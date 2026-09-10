"""vLLM batch logits processor for SSBD biased draft verification."""

from __future__ import annotations

from typing import Any, Sequence

SSBD_EXTRA_DRAFT = "ssbd_draft_token_ids"
SSBD_EXTRA_BOOST = "ssbd_logit_boost"

SSBD_VLLM_LOGITS_PROCESSOR_AVAILABLE = False
SSBDDraftBiasLogitsProcessor: Any = None

try:
    import torch
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import LogitsProcessor
    from vllm.v1.sample.logits_processor.interface import (
        BatchUpdate,
        MoveDirectionality,
    )

    def _draft_prefix_matches(
        output_tok_ids: Sequence[int], draft_ids: Sequence[int], step: int
    ) -> bool:
        if step <= 0:
            return True
        if len(output_tok_ids) < step:
            return False
        return list(output_tok_ids[:step]) == list(draft_ids[:step])

    class SSBDDraftBiasLogitsProcessor(LogitsProcessor):
        """Boost the current draft token during SSBD verify steps only (paper beta)."""

        @classmethod
        def validate_params(cls, params: SamplingParams) -> None:
            extra = params.extra_args or {}
            draft = extra.get(SSBD_EXTRA_DRAFT)
            if draft is None:
                return
            if not isinstance(draft, list) or not all(isinstance(x, int) for x in draft):
                raise ValueError(f"{SSBD_EXTRA_DRAFT} must be a list of ints")
            boost = extra.get(SSBD_EXTRA_BOOST, 0.0)
            if boost is not None and not isinstance(boost, (int, float)):
                raise ValueError(f"{SSBD_EXTRA_BOOST} must be numeric")

        def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
            del vllm_config, device, is_pin_memory
            self.req_info: dict[int, tuple[list[int], float, list[int]]] = {}

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None) -> None:
            if batch_update is None:
                return

            for idx in batch_update.removed:
                self.req_info.pop(idx, None)

            for idx, params, _prompt, output_tok_ids in batch_update.added:
                extra = params.extra_args or {}
                draft = extra.get(SSBD_EXTRA_DRAFT)
                boost = float(extra.get(SSBD_EXTRA_BOOST, 0.0) or 0.0)
                if (
                    isinstance(draft, list)
                    and draft
                    and boost > 0.0
                    and all(isinstance(token_id, int) for token_id in draft)
                ):
                    self.req_info[idx] = (draft, boost, output_tok_ids)
                else:
                    self.req_info.pop(idx, None)

            for src_idx, dst_idx, direction in batch_update.moved:
                src_val = self.req_info.pop(src_idx, None)
                dst_val = self.req_info.pop(dst_idx, None)
                if src_val is not None:
                    self.req_info[dst_idx] = src_val
                if direction == MoveDirectionality.SWAP and dst_val is not None:
                    self.req_info[src_idx] = dst_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            for idx, (draft_ids, boost, output_tok_ids) in self.req_info.items():
                if idx >= logits.shape[0] or boost <= 0.0 or not draft_ids:
                    continue
                step = len(output_tok_ids)
                if step >= len(draft_ids):
                    continue
                if not _draft_prefix_matches(output_tok_ids, draft_ids, step):
                    continue
                draft_id = draft_ids[step]
                if 0 <= draft_id < logits.shape[1]:
                    logits[idx, draft_id] += boost
            return logits

    SSBD_VLLM_LOGITS_PROCESSOR_AVAILABLE = True
except ImportError:
    pass
