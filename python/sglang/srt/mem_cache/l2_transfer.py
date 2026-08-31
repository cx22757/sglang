from __future__ import annotations

import logging
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    kwargs = {"enable_timing": True} if timing_enabled else {}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), timing_enabled


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            debug_sources = self._debug_snapshots(transfers, from_host=False)
            ack_start.record()
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
            ack_finish.record()
            if debug_sources is not None:
                ack_finish.synchronize()
                self._log_debug_copies(
                    "D2H", transfers, debug_sources, destination_from_host=True
                )
            self._record_stream(transfers, self.device_to_host_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        layer_num: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        primary = transfers[0] if transfers else None
        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            debug_sources = self._debug_snapshots(transfers, from_host=True)
            ack_start.record()
            for layer_id in range(layer_num):
                for transfer in transfers:
                    local_layer_id = (
                        transfer.layer_mapper(layer_id)
                        if transfer.layer_mapper is not None
                        else layer_id
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and layer_id >= transfer.host_pool.layer_num
                    ):
                        continue
                    transfer.host_pool.load_to_device_per_layer(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
            ack_finish.record()
            if debug_sources is not None:
                ack_finish.synchronize()
                self._log_debug_copies(
                    "H2D", transfers, debug_sources, destination_from_host=False
                )
            self._record_stream(transfers, self.host_to_device_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    @staticmethod
    def _debug_snapshots(transfers: list[L2Transfer], *, from_host: bool):
        if not envs.SGLANG_DSV4_KV_DIGEST.get():
            return None
        snapshots = []
        for transfer in transfers:
            digest_fn = getattr(transfer.host_pool, "debug_transfer_digest", None)
            indices = transfer.host_indices if from_host else transfer.device_indices
            if digest_fn is None:
                snapshots.append(None)
                continue
            try:
                snapshots.append(digest_fn(indices, from_host=from_host))
            except Exception as exc:  # noqa: BLE001
                snapshots.append((0, f"ERR:{type(exc).__name__}", "error"))
        return snapshots

    @classmethod
    def _log_debug_copies(
        cls,
        direction: str,
        transfers: list[L2Transfer],
        sources,
        *,
        destination_from_host: bool,
    ) -> None:
        destinations = cls._debug_snapshots(
            transfers, from_host=destination_from_host
        )
        for transfer, source, destination in zip(transfers, sources, destinations):
            if source is None or destination is None:
                continue
            src_bytes, src_digest, src_rows = source
            dst_bytes, dst_digest, dst_rows = destination
            matched = (
                not src_digest.startswith("ERR:")
                and not dst_digest.startswith("ERR:")
                and src_bytes == dst_bytes
                and src_digest == dst_digest
            )
            logger.warning(
                "[KVCOPY] direction=%s pool=%s pages=%d "
                "src_rows=%s dst_rows=%s bytes=%d src=%s dst=%s match=%d",
                direction,
                getattr(transfer.host_pool, "pool_name", "unknown"),
                len(transfer.host_indices)
                // max(1, getattr(transfer.host_pool, "page_size", 1)),
                src_rows,
                dst_rows,
                dst_bytes,
                src_digest,
                dst_digest,
                int(matched),
            )

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.is_cuda:
                    indices.record_stream(stream)
