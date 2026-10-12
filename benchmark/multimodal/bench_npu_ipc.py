"""Sequential IPC transport probe using real Qwen-VL processor outputs.

Run inside an Ascend container. This is a transport microbenchmark, not an
SGLang integration or a measurement of request TTFT. A producer waits for the
consumer after every transfer, so one reusable buffer has a safe lifetime.
"""

import argparse
import json
import multiprocessing as mp
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def consumer(connection, device):
    try:
        torch.npu.set_device(device)
        connection.send({"ready": True})
        while True:
            setup = connection.recv()
            if setup is None:
                return
            mode, shape, dtype, payload = setup
            destination = torch.empty(shape, dtype=dtype, device=f"npu:{device}")
            shared = None
            if mode == "cpu_shm":
                shared = payload
            elif mode in ("ipc_copy", "ipc_borrow"):
                rebuild, arguments = payload
                arguments = list(arguments)
                arguments[6] = device
                shared = rebuild(*arguments)
            torch.npu.synchronize()
            connection.send({"setup": True})
            while True:
                command = connection.recv()
                if command is None:
                    break
                start = time.perf_counter()
                if mode == "cpu_shm":
                    destination.copy_(shared)
                    tensor = destination
                elif mode == "ipc_copy":
                    destination.copy_(shared)
                    tensor = destination
                else:
                    tensor = shared
                torch.npu.synchronize()
                transport_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                # Read all the feature bytes even in the borrowed-view case.
                checksum = tensor.float().sum().item()
                torch.npu.synchronize()
                read_ms = (time.perf_counter() - start) * 1000
                connection.send(
                    {
                        "transport_ms": transport_ms,
                        "read_ms": read_ms,
                        "checksum": checksum,
                    }
                )
            del shared, destination, tensor
            connection.send({"finished": True})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise


def receive(connection, timeout):
    if not connection.poll(timeout):
        raise TimeoutError("Consumer did not respond within the timeout")
    message = connection.recv()
    if "error" in message:
        raise RuntimeError(message["error"])
    return message


def summary(samples):
    values = sorted(samples)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
    }


def measure(connection, source, mode, args):
    from torch.multiprocessing.reductions import reduce_tensor

    if mode == "cpu_shm":
        buffer = torch.empty(source.shape, dtype=source.dtype).share_memory_()
        payload = buffer
    else:
        buffer = torch.empty_like(source)
        payload = reduce_tensor(buffer)
    start = time.perf_counter()
    connection.send((mode, source.shape, source.dtype, payload))
    receive(connection, args.timeout)
    setup_ms = (time.perf_counter() - start) * 1000
    expected = source.float().sum().item()
    samples = {name: [] for name in ("producer", "consumer", "read", "roundtrip")}
    for step in range(args.warmup + args.iterations):
        start = time.perf_counter()
        buffer.copy_(source)
        torch.npu.synchronize()
        producer_ms = (time.perf_counter() - start) * 1000
        connection.send(True)
        message = receive(connection, args.timeout)
        roundtrip_ms = (time.perf_counter() - start) * 1000
        if abs(message["checksum"] - expected) > max(1e-3, abs(expected) * 1e-5):
            raise AssertionError(f"Feature checksum mismatch: {message} != {expected}")
        if step >= args.warmup:
            samples["producer"].append(producer_ms)
            samples["consumer"].append(message["transport_ms"])
            samples["read"].append(message["read_ms"])
            samples["roundtrip"].append(roundtrip_ms)
    connection.send(None)
    receive(connection, args.timeout)
    return {
        "mode": mode,
        "setup_ms": setup_ms,
        **{key: summary(values) for key, values in samples.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/paas/weights/Qwen3-VL-4B-Instruct")
    parser.add_argument("--producer-device", type=int, default=0)
    parser.add_argument("--consumer-device", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 448, 896, 1344])
    parser.add_argument("--images", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["cpu_shm", "ipc_copy", "ipc_borrow"],
        choices=["cpu_shm", "ipc_copy", "ipc_borrow"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be nonnegative")
    from PIL import Image
    from transformers import AutoProcessor

    from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
        npu_apply_qwen_image_preprocess_patch,
    )

    torch.npu.set_device(args.producer_device)
    npu_apply_qwen_image_preprocess_patch()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=consumer, args=(child, args.consumer_device))
    process.start()
    child.close()
    results = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "processor": type(processor.image_processor).__name__,
        "arguments": {**vars(args), "output": str(args.output)},
        "cases": [],
    }
    try:
        receive(parent, args.timeout)
        for size in args.sizes:
            start = time.perf_counter()
            output = processor.image_processor(
                images=[Image.new("RGB", (size, size), (80, 120, 160))],
                device=f"npu:{args.producer_device}",
                return_tensors="pt",
            )
            features = output["pixel_values"]
            torch.npu.synchronize()
            preprocess_ms = (time.perf_counter() - start) * 1000
            if features.device.type != "npu":
                raise RuntimeError(f"Expected device features, got {features.device}")
            for count in args.images:
                source = features.repeat(count, 1).contiguous()
                case = {
                    "image_size": size,
                    "images": count,
                    "shape": list(source.shape),
                    "dtype": str(source.dtype),
                    "feature_mib": source.numel() * source.element_size() / 2**20,
                    "single_image_preprocess_ms": preprocess_ms,
                    "grid": output["image_grid_thw"].tolist(),
                    "measurements": [],
                }
                for mode in args.modes:
                    measurement = measure(parent, source, mode, args)
                    case["measurements"].append(measurement)
                    print(
                        json.dumps({**case, "measurements": [measurement]}), flush=True
                    )
                results["cases"].append(case)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
        parent.send(None)
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"Consumer exit code: {process.exitcode}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent.close()


if __name__ == "__main__":
    main()
