"""Opt-in, real-checkpoint audit of GLM layer-0 mHC and QKV loading."""

import json
import logging
from collections import Counter

import torch

logger = logging.getLogger(__name__)


class Glm5WeightLoadingAudit:
    def __init__(self, params, rank, tp_size):
        self.params = params
        self.rank = rank
        self.tp_size = tp_size
        self.prefix = "model.layers.0."
        self.hc_target = self.prefix + "hc_attn_fn"
        self.qkv_target = self.prefix + "self_attn.qkv_proj.weight"
        self.targets = (self.hc_target, self.qkv_target)
        self.sources = {
            "model.language_model.layers.0.attn_hc.fn": (self.hc_target, None),
            **{
                f"model.language_model.layers.0.self_attn.{shard}_proj.weight": (
                    self.qkv_target,
                    shard,
                )
                for shard in ("q", "k", "v")
            },
        }
        missing = [name for name in self.targets if name not in params]
        if missing:
            raise RuntimeError(f"GLM weight audit missing targets: {missing}")
        self.references = {}
        self.source_counts = Counter()
        self.events = []

    def observe_sources(self, weights):
        for source, tensor in weights:
            if source in self.sources:
                target, shard = self.sources[source]
                self.source_counts[source] += 1
                # Derive the reference independently of the production loader.
                # Layer-0 KDA has equal Q/K/V heads; all are output-sharded.
                reference = tensor
                rows = None
                if shard is not None:
                    assert tensor.shape[0] % self.tp_size == 0
                    size = tensor.shape[0] // self.tp_size
                    start = self.rank * size
                    rows = [start, start + size]
                    reference = tensor[start : start + size]
                self.references[(target, shard)] = reference.detach().to(
                    device="cpu", dtype=self.params[target].dtype, copy=True
                )
                logger.info(
                    "GLM_WEIGHT_AUDIT_SOURCE %s",
                    json.dumps(
                        dict(
                            rank=self.rank,
                            source=source,
                            target=target,
                            shard=shard,
                            source_shape=list(tensor.shape),
                            source_dtype=str(tensor.dtype),
                            target_shape=list(self.params[target].shape),
                            target_dtype=str(self.params[target].dtype),
                            source_rows=rows,
                        )
                    ),
                )
            yield source, tensor

    def record_loaded(self, target, shard=None):
        if target in self.targets:
            self.events.append((target, shard))

    def finish(self):
        expected_keys = set(self.sources.values())
        event_counts = Counter(self.events)
        coverage_ok = (
            set(self.references) == expected_keys
            and set(event_counts) == expected_keys
            and all(count == 1 for count in self.source_counts.values())
            and all(count == 1 for count in event_counts.values())
        )
        comparisons = []
        if set(self.references) == expected_keys:
            expected = {
                self.hc_target: self.references[(self.hc_target, None)],
                self.qkv_target: torch.cat(
                    [self.references[(self.qkv_target, s)] for s in ("q", "k", "v")],
                    dim=0,
                ),
            }
            for target, reference in expected.items():
                actual = self.params[target].detach().cpu()
                same_shape = actual.shape == reference.shape
                finite = bool(torch.isfinite(actual).all()) and bool(
                    torch.isfinite(reference).all()
                )
                equal = same_shape and finite and torch.equal(actual, reference)
                comparisons.append(
                    dict(
                        target=target,
                        shape=list(actual.shape),
                        dtype=str(actual.dtype),
                        finite=finite,
                        equal=equal,
                        max_abs=(actual.float() - reference.float()).abs().max().item()
                        if same_shape and finite
                        else None,
                    )
                )
        passed = coverage_ok and all(c["equal"] for c in comparisons)
        logger.info(
            "GLM_WEIGHT_AUDIT_RESULT %s",
            json.dumps(
                dict(
                    rank=self.rank,
                    tp_size=self.tp_size,
                    scope="layer0_hc_attn_fn_and_qkv_before_post_load",
                    source_counts=dict(self.source_counts),
                    successful_load_events=self.events,
                    coverage_ok=coverage_ok,
                    comparisons=comparisons,
                    passed=passed,
                )
            ),
        )
        if not passed:
            raise RuntimeError("GLM layer-0 weight loading audit failed; see audit log")
