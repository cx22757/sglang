"""Unit tests for the Ascend DSV4 Compressor explicit-state ABI."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="base-a-test-1-npu-a2")

for mod in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sgl_kernel_npu.norm",
    "sgl_kernel_npu.norm.add_rmsnorm_bias",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(mod, MagicMock())

deepseek_v2_stub = ModuleType("sglang.srt.models.deepseek_v2")
deepseek_v2_stub._is_hip = False
sys.modules.setdefault("sglang.srt.models.deepseek_v2", deepseek_v2_stub)

from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (  # noqa: E402
    _build_explicit_state_block_table,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (  # noqa: E402
    DeepSeekV4TokenToKVPool,
)


class TestExplicitStateBlockTable(unittest.TestCase):
    def test_c4_maps_history_to_explicit_swa_state_locations(self):
        class StatePool:
            dummy_state_loc = 99

            @staticmethod
            def translate_from_swa_loc_to_state_loc(loc):
                return loc

        token_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda loc: loc,
        )
        table = _build_explicit_state_block_table(
            compress_ratio=4,
            coff=2,
            state_pool=StatePool(),
            token_to_kv_pool=token_pool,
            req_to_token=torch.arange(24, dtype=torch.int64).view(1, -1),
            req_pool_indices=torch.tensor([0]),
            start_pos=torch.tensor([4], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            seqused=torch.tensor([2], dtype=torch.int32),
            max_input_capacity=8,
        )
        self.assertEqual(table.tolist(), [[99] * 4 + list(range(6)) + [99] * 6])

    def test_c128_uses_explicit_request_state_locations(self):
        class StatePool:
            dummy_state_loc = 777

            @staticmethod
            def translate_from_req_position_to_state_loc(reqs, positions):
                return reqs * 128 + positions % 128

        table = _build_explicit_state_block_table(
            compress_ratio=128,
            coff=1,
            state_pool=StatePool(),
            token_to_kv_pool=SimpleNamespace(),
            req_to_token=torch.zeros((4, 384), dtype=torch.int64),
            req_pool_indices=torch.tensor([2]),
            start_pos=torch.tensor([2], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            seqused=torch.tensor([2], dtype=torch.int32),
            max_input_capacity=4,
        )
        self.assertEqual(table.shape, (1, 132))
        self.assertEqual(table[0, :126].tolist(), [777] * 126)
        self.assertEqual(table[0, 126:130].tolist(), [256, 257, 258, 259])
        self.assertEqual(table[0, 130:].tolist(), [777, 777])

    def test_c128_request_teardown_clears_unshifted_state_bank(self):
        state = torch.ones((4 * 128, 4), dtype=torch.float32)
        pool = SimpleNamespace(
            ratio=128,
            online=False,
            ring_size=128,
            kv_score_buffer=SimpleNamespace(kv_score=state),
        )
        token_pool = SimpleNamespace(compress_state_pools=[pool])

        DeepSeekV4TokenToKVPool.clear_c128_req_state(token_pool, req_pool_idx=1)

        self.assertTrue(torch.equal(state[:128], torch.ones_like(state[:128])))
        cleared = state[128 : 2 * 128]
        self.assertTrue(torch.equal(cleared[:, :2], torch.zeros_like(cleared[:, :2])))
        self.assertTrue(torch.isneginf(cleared[:, 2:]).all())
        self.assertTrue(
            torch.equal(state[2 * 128 :], torch.ones_like(state[2 * 128 :]))
        )

    def test_pool_flush_clears_every_c128_request_state_bank(self):
        class StateBuffer:
            def __init__(self):
                self.kv_score = torch.ones((4 * 128, 4), dtype=torch.float32)

            def clear(self):
                half = self.kv_score.shape[-1] // 2
                self.kv_score[:, :half].zero_()
                self.kv_score[:, half:].fill_(float("-inf"))

        buffer = StateBuffer()
        pool = SimpleNamespace(ratio=128, kv_score_buffer=buffer)
        token_pool = SimpleNamespace(compress_state_pools=[pool])

        DeepSeekV4TokenToKVPool.clear_all_c128_req_states(token_pool)

        self.assertTrue(
            torch.equal(
                buffer.kv_score[:, :2], torch.zeros_like(buffer.kv_score[:, :2])
            )
        )
        self.assertTrue(torch.isneginf(buffer.kv_score[:, 2:]).all())

if __name__ == "__main__":
    unittest.main()
