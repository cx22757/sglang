"""Eager Ascend KPool reference path using allocator-global index cache slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compress_index_keys(keys, scores, ape):
    """Per-channel softmax over the four consecutive keys, before Hadamard."""
    return (keys.float() * (scores.float() + ape.float()).softmax(dim=1)).sum(1)


def pooled_causal_topk(query, pooled_keys, weights, positions, topk, kpool):
    """Return logical FULL token indices, including each query's incomplete tail."""
    rows = query.shape[0]
    result = torch.full(
        (rows, topk + kpool - 1), -1, dtype=torch.int32, device=query.device
    )
    completed = torch.div(positions + 1, kpool, rounding_mode="floor")
    group_budget = topk // kpool
    num_groups = pooled_keys.shape[0]
    # Bound the temporary [query, head, history-group] allocation.
    for start in range(0, rows, 32):
        end = min(rows, start + 32)
        if num_groups:
            scores = torch.einsum(
                "qhd,gd->qhg", query[start:end].float(), pooled_keys.float()
            ).relu()
            scores = (scores * weights[start:end, :, None]).sum(1)
            group_ids = torch.arange(num_groups, device=query.device)
            valid = group_ids[None, :] < completed[start:end, None]
            scores.masked_fill_(~valid, float("-inf"))
            count = min(group_budget, num_groups)
            selected = scores.topk(count, dim=-1, sorted=False).indices
            selected_valid = selected < completed[start:end, None]
            offsets = torch.arange(kpool, device=query.device)
            tokens = selected[..., None] * kpool + offsets
            tokens.masked_fill_(~selected_valid[..., None], -1)
            result[start:end, : count * kpool] = tokens.flatten(1).to(torch.int32)
        tail_offsets = torch.arange(kpool - 1, device=query.device)
        tail = completed[start:end, None] * kpool + tail_offsets
        tail.masked_fill_(tail > positions[start:end, None], -1)
        result[start:end, topk:] = tail.to(torch.int32)
    return result


def forward_kpool_npu(
    indexer, x, q_lora, positions, forward_batch, layer_id, return_indices
):
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode():
        raise NotImplementedError(
            "Ascend GLM5 KPool reference path requires eager execution"
        )
    mode = forward_batch.forward_mode
    if not (mode.is_extend_without_speculative() or mode.is_decode_or_idle()):
        raise NotImplementedError(
            "Ascend GLM5 KPool currently supports eager prefill/decode"
        )
    if forward_batch.attn_cp_metadata is not None:
        raise NotImplementedError(
            "Ascend GLM5 KPool does not yet support context parallelism"
        )
    if x.shape[0] == 0 or mode.is_idle():
        return (
            torch.full(
                (x.shape[0], indexer.index_topk + indexer.index_kpool - 1),
                -1,
                dtype=torch.int32,
                device=x.device,
            )
            if return_indices
            else None
        )

    pool = get_token_to_kv_pool()
    if hasattr(pool, "full_kv_pool"):
        layer_id = pool.full_attention_layer_id_mapping[layer_id]
        pool = pool.full_kv_pool
    if pool.index_kpool != indexer.index_kpool or pool.kpool_gate_buffer is None:
        raise RuntimeError("Ascend KPool requires a matching raw-key/gate KV cache")
    if pool.index_k_scale_buffer is not None:
        raise NotImplementedError(
            "Ascend GLM5 KPool reference cache requires BF16 index keys"
        )
    query = indexer.wq_b(q_lora)[0].view(-1, indexer.n_heads, indexer.head_dim)
    key = indexer.k_norm(indexer.wk(x)[0])
    if not indexer.skip_rope and indexer.rope_head_dim:
        q_rope, k_rope = indexer.rotary_emb(
            positions,
            query[..., : indexer.rope_head_dim],
            key[..., : indexer.rope_head_dim],
        )
        query = torch.cat((q_rope, query[..., indexer.rope_head_dim :]), dim=-1)
        key = torch.cat((k_rope, key[..., indexer.rope_head_dim :]), dim=-1)
    # The GPU rotates both pooled keys and queries by the same normalized
    # Hadamard matrix. Dot products are invariant; BF16 reference uses neither.
    gate = F.linear(x, indexer.index_kpool_compress_gate)
    pool.set_index_k_buffer(layer_id, forward_batch.out_cache_loc, key)
    pool.set_kpool_gate_buffer(layer_id, forward_batch.out_cache_loc, gate)
    if not return_indices:
        return None
    weights = indexer.weights_proj(x.float())[0]
    weights = weights * indexer.n_heads**-0.5 * indexer.softmax_scale
    raw_keys = pool.get_index_k_buffer(layer_id).view(-1, indexer.head_dim)
    gate_cache = pool.get_kpool_gate_buffer(layer_id).view(-1, indexer.head_dim)
    req_table = get_req_to_token_pool().req_to_token
    lengths = forward_batch.seq_lens.detach().cpu().tolist()
    query_lengths = (
        forward_batch.extend_seq_lens_cpu
        if mode.is_extend_without_speculative()
        else [1] * len(lengths)
    )
    requests = forward_batch.req_pool_indices.detach().cpu().tolist()
    output = []
    offset = 0
    for req, seq_len, q_len in zip(requests, lengths, query_lengths):
        q_len = int(q_len)
        num_groups = int(seq_len) // indexer.index_kpool
        locs = req_table[req, : num_groups * indexer.index_kpool].long()
        shape = (num_groups, indexer.index_kpool, indexer.head_dim)
        pooled = compress_index_keys(
            raw_keys[locs].view(shape),
            gate_cache[locs].view(shape),
            indexer.index_kpool_compress_ape,
        )
        output.append(
            pooled_causal_topk(
                query[offset : offset + q_len],
                pooled,
                weights[offset : offset + q_len],
                positions[offset : offset + q_len],
                indexer.index_topk,
                indexer.index_kpool,
            )
        )
        offset += q_len
    return torch.cat(output, dim=0)
