"""Unit tests for the Ascend DSV4 Compressor paged-state ABI."""

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
    _COMPRESSOR_CACHE_MODE,
    _build_paged_state_block_table,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (  # noqa: E402
    DeepSeekV4TokenToKVPool,
)


class TestPagedStateBlockTable(unittest.TestCase):
    def test_uses_cache_mode_supported_by_ascend_operator(self):
        self.assertEqual(_COMPRESSOR_CACHE_MODE, 1)

    def test_c4_maps_global_blocks_to_shifted_swa_state_blocks(self):
        class StatePool:
            page_size = 8

            @staticmethod
            def translate_from_swa_loc_to_state_loc(loc):
                return (loc // 8 + 1) * 8 + loc % 8

        token_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda loc: loc,
        )
        table = _build_paged_state_block_table(
            compress_ratio=4,
            coff=2,
            state_pool=StatePool(),
            token_to_kv_pool=token_pool,
            req_to_token=torch.arange(24, dtype=torch.int64).view(1, -1),
            req_pool_indices=torch.tensor([0]),
            start_pos=torch.tensor([8], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 8], dtype=torch.int32),
            seqused=torch.tensor([8], dtype=torch.int32),
            max_input_capacity=8,
        )
        self.assertEqual(table.tolist(), [[1, 2, 0]])

    def test_c128_reuses_positive_request_state_block(self):
        class StatePool:
            page_size = 128

            @staticmethod
            def translate_from_req_position_to_state_loc(reqs, positions):
                return (reqs + 1) * 128 + positions % 128

        table = _build_paged_state_block_table(
            compress_ratio=128,
            coff=1,
            state_pool=StatePool(),
            token_to_kv_pool=SimpleNamespace(),
            req_to_token=torch.zeros((4, 384), dtype=torch.int64),
            req_pool_indices=torch.tensor([2]),
            start_pos=torch.tensor([128], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 128], dtype=torch.int32),
            seqused=torch.tensor([128], dtype=torch.int32),
            max_input_capacity=128,
        )
        self.assertEqual(table.tolist(), [[3, 3, 0]])

    def test_c128_request_teardown_clears_shifted_state_bank(self):
        state = torch.ones((4 * 128, 4), dtype=torch.float32)
        pool = SimpleNamespace(
            ratio=128,
            online=False,
            ring_size=128,
            state_page_offset=1,
            kv_score_buffer=SimpleNamespace(kv_score=state),
        )
        token_pool = SimpleNamespace(compress_state_pools=[pool])

        DeepSeekV4TokenToKVPool.clear_c128_req_state(token_pool, req_pool_idx=1)

        self.assertTrue(torch.equal(state[: 2 * 128], torch.ones_like(state[: 2 * 128])))
        cleared = state[2 * 128 : 3 * 128]
        self.assertTrue(torch.equal(cleared[:, :2], torch.zeros_like(cleared[:, :2])))
        self.assertTrue(torch.isneginf(cleared[:, 2:]).all())
        self.assertTrue(torch.equal(state[3 * 128 :], torch.ones_like(state[3 * 128 :])))

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

        self.assertTrue(torch.equal(buffer.kv_score[:, :2], torch.zeros_like(buffer.kv_score[:, :2])))
        self.assertTrue(torch.isneginf(buffer.kv_score[:, 2:]).all())

if __name__ == "__main__":
    unittest.main()
