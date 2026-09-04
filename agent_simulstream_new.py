import logging
import math
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from openai import OpenAI
from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput
from vllm import LLM, SamplingParams

from streaming_sfm.hyp_utils import (
    HoldNHypothesisBuffer,
    LACPHypothesisBuffer,
    LCPHypothesisBuffer,
    WaitKHypothesisBuffer,
)
from streaming_sfm.parakeet import _build_slcp_buffer
from streaming_sfm import LOG_LEVEL
from streaming_sfm.streaming_model import (
    StreamingBatchedAudioBufferWithOffset,
    StreamingParakeet,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)

# Default MT checkpoint when `llm_model_name` is omitted from config (vLLM / OpenAI-compatible).
QWEN35_4B_MODEL_NAME = "Qwen/Qwen3.5-4B"
QWEN35_9B_MODEL_NAME = "Qwen/Qwen3.5-9B"
QWEN35_27B_MODEL_NAME = "Qwen/Qwen3.5-27B"
DEFAULT_LLM_MODEL_NAME = QWEN35_4B_MODEL_NAME

# vLLM ``quantization`` values we accept via ``llm_quantization`` (plus aliases below).
_LLM_QUANT_ALIASES = {
    "4bit": "bitsandbytes",
    "bnb4": "bitsandbytes",
    "bnb_4bit": "bitsandbytes",
    "bitsandbytes": "bitsandbytes",
    "bnb": "bitsandbytes",
    "8bit": "bitsandbytes",
    "bnb8": "bitsandbytes",
    "bnb_8bit": "bitsandbytes",
    "awq": "awq",
    "gptq": "gptq",
    "gptq_marlin": "gptq_marlin",
    "awq_marlin": "awq_marlin",
    "compressed-tensors": "compressed-tensors",
    "compressed_tensors": "compressed-tensors",
}
_LLM_QUANT_8BIT_ALIASES = frozenset({"8bit", "bnb8", "bnb_8bit"})


def _is_qwen35_model(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", ".")
    return "qwen3.5" in normalized


def _normalize_llm_dtype(dtype: str) -> str:
    """Map config aliases to vLLM ``dtype`` values."""
    key = dtype.lower().strip()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "half": "half",
        "fp32": "float32",
        "float32": "float32",
        "float": "float",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "auto": "auto",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported llm_dtype={dtype!r}; use one of {sorted(aliases)}"
        )
    return aliases[key]


def _normalize_llm_quantization(quant: str) -> str:
    """Map config aliases to vLLM ``quantization`` method names."""
    key = quant.lower().strip()
    if key not in _LLM_QUANT_ALIASES:
        raise ValueError(
            f"Unsupported llm_quantization={quant!r}; "
            f"use one of {sorted(_LLM_QUANT_ALIASES)}"
        )
    return _LLM_QUANT_ALIASES[key]


# vLLM EngineArgs ``quantization_config`` is only for online quant schemes (v0.20+).
# Bitsandbytes inflight 4-bit uses ``quantization="bitsandbytes"`` alone; vLLM applies
# BitsAndBytesConfig defaults (load_in_4bit=True) via the model loader.
_LLM_ONLINE_QUANTIZATION = frozenset(
    {
        "fp8_per_block",
        "fp8_per_tensor",
        "int8_per_channel_weight_only",
        "mxfp8",
        "online",
    }
)


def _resolve_llm_quantization(
    config: SimpleNamespace,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Return (vLLM quantization method, optional EngineArgs quantization_config).

    ``llm_quantization_config`` is only forwarded for online vLLM quant schemes.
    Bitsandbytes (``4bit`` / ``bitsandbytes``) must not use EngineArgs
    ``quantization_config`` — vLLM 0.20 rejects it and uses loader defaults instead.
    """
    explicit = getattr(config, "llm_quantization", None)
    if explicit is None:
        return None, None

    method = _normalize_llm_quantization(explicit)
    raw_cfg = getattr(config, "llm_quantization_config", None)
    if raw_cfg is not None:
        if method == "bitsandbytes":
            logger.warning(
                "llm_quantization_config is ignored for bitsandbytes in vLLM; "
                "use a pre-quantized HF checkpoint or hf_overrides instead."
            )
        elif method in _LLM_ONLINE_QUANTIZATION:
            if hasattr(raw_cfg, "items"):
                return method, dict(raw_cfg)
            raise TypeError(
                "llm_quantization_config must be a mapping (dict / OmegaConf DictConfig)"
            )
        else:
            logger.warning(
                "llm_quantization_config is not passed to vLLM for quantization=%s",
                method,
            )
    if method == "bitsandbytes" and (
        getattr(config, "llm_load_in_8bit", False)
        or explicit.lower().strip() in _LLM_QUANT_8BIT_ALIASES
    ):
        logger.warning(
            "8-bit bitsandbytes is not configurable via EngineArgs in this vLLM "
            "version; inflight loading defaults to 4-bit."
        )
    return method, None


def _resolve_llm_dtype(
    config: SimpleNamespace,
    llm_model_name: str,
    llm_quantization: Optional[str],
) -> Optional[str]:
    """Return vLLM dtype; fp16 default for 9B only when not using weight quantization."""
    explicit = getattr(config, "llm_dtype", None)
    if explicit is not None:
        return _normalize_llm_dtype(explicit)
    if llm_quantization is not None:
        # vLLM bitsandbytes recipe uses bf16 activations; weights are 4/8-bit.
        return "bfloat16"
    if "9b" in llm_model_name.lower():
        return "float16"
    return None

# Clause/sentence punctuation immediately followed by a word (any Unicode letter).
_PUNCT_GLUE_RE = re.compile(r"([,;:.!?])(\S)")
# Model filler / thinking artifacts: "...", "…", or suffixes like "fait...".
_ELLIPSIS_RE = re.compile(r"…|\.{2,}")


def _starts_word_char(ch: str) -> bool:
    """True when ch begins a new word (letters incl. É; not digits/punctuation)."""
    return unicodedata.category(ch).startswith("L")


def insert_space_after_punctuation(text: str) -> str:
    """
    Fix model-glued boundaries such as ``Sozialversicherung.Jede`` or ``PCNode.Émos``.

    Skips digits after ``.`` so values like ``3.14`` / ``22.000`` stay intact.
    """
    if not text:
        return text

    def repl(match: re.Match) -> str:
        following = match.group(2)
        if _starts_word_char(following[0]):
            return f"{match.group(1)} {following}"
        return match.group(0)

    return _PUNCT_GLUE_RE.sub(repl, text)


def strip_ellipsis(text: str) -> str:
    """Remove Unicode or ASCII ellipsis runs (e.g. ``...``, ``fait...``)."""
    if not text:
        return text
    return _ELLIPSIS_RE.sub("", text)


def longest_common_prefix(s1: str, s2: str) -> str:
    t1 = s1.split()
    t2 = s2.split()
    for i in range(min(len(t1), len(t2))):
        if t1[i] != t2[i]:
            return ' '.join(t1[:i])
    return ' '.join(t1[: min(len(t1), len(t2))])


# MT look-ahead modes (EAC/PAC lives in agent_simulstream.py; SSBD is reserved).
MT_LOOKAHEAD_NONE = "none"
MT_LOOKAHEAD_EAC = "eac"
MT_LOOKAHEAD_TAF = "taf"
MT_LOOKAHEAD_SSBD = "ssbd"
MT_LOOKAHEAD_MODES = frozenset(
    {MT_LOOKAHEAD_NONE, MT_LOOKAHEAD_EAC, MT_LOOKAHEAD_TAF, MT_LOOKAHEAD_SSBD}
)


def _truncate_to_max_words(text: str, max_words: int) -> str:
    if max_words <= 0 or not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words])


def majority_vote_prefix(hypotheses: Sequence[str], agree_thres: float) -> str:
    """
    Return the longest word prefix agreed on by at least ``agree_thres`` fraction
    of hypotheses (TAF-style source anticipation).
    """
    cleaned = [h.strip() for h in hypotheses if h and h.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    tokenized = [h.split() for h in cleaned]
    max_len = max(len(tokens) for tokens in tokenized)
    required = max(1, math.ceil(len(cleaned) * agree_thres))

    agreed: List[str] = []
    for idx in range(max_len):
        counts: dict[str, int] = {}
        for tokens in tokenized:
            if idx >= len(tokens):
                continue
            word = tokens[idx]
            counts[word] = counts.get(word, 0) + 1
        if not counts:
            break
        best_word = max(counts, key=counts.get)
        if counts[best_word] >= required:
            agreed.append(best_word)
        else:
            break
    return " ".join(agreed)


@dataclass
class CascadeState:
    speech_id: int = 0
    asr_committed_text: str = ""
    prev_translation: str = ""
    translation_hypotheses: List[str] = field(default_factory=lambda: [""])
    full_mt_hypothesis: str = ""
    mt_llm_calls_step: int = 0
    mt_llm_calls_utterance: int = 0
    ssbd_draft_accepted_tokens: int = 0
    ssbd_draft_total_tokens: int = 0
    ssbd_verify_steps: int = 0
    ssbd_verify_steps_zero_accept: int = 0
    ssbd_cumulative_draft_tokens: int = 0
    ssbd_cumulative_accepted_tokens: int = 0
    displayed_hypothesis: str = ""
    pending_new_tokens: List[str] = field(default_factory=list)
    pending_deleted_tokens: List[str] = field(default_factory=list)
    emission_started: bool = False
    total_samples: int = 0

    # Streaming ASR internals
    asr_buffer: Optional[StreamingBatchedAudioBufferWithOffset] = None
    asr_hyp_buffer: Optional[object] = None
    current_offset: int = 0
    nchunks_no_output: int = 0
    consecutive_empty_mt: int = 0


class CascadeSpeechProcessor(SpeechProcessor):
    """
    SimulStream processor with:
    - ASR: Streaming SFM + Parakeet
    - MT: Qwen3.5 (e.g. 4B / 9B / 27B) via vLLM (local or OpenAI-compatible endpoint)

    MT look-ahead modes (``mt_lookahead_mode``):
    - ``none``: committed ASR only (default)
    - ``taf``: TAF-style source anticipation via extra LLM calls
    - ``ssbd``: self-speculative biased decoding (Zeng et al., 2026)
      Set ``ssbd_raw_emission: true`` to emit full retranslations with deletions
      (paper-style NE) instead of LCP-stable append-only output.
    - ``eac``: use ``agent_simulstream.py`` (PAC / provisional ASR context)
    """

    @staticmethod
    def _resolve_llm_max_num_seqs(config: SimpleNamespace) -> int:
        explicit = getattr(config, "llm_max_num_seqs", None)
        if explicit is not None:
            return max(1, int(explicit))
        mode = getattr(config, "mt_lookahead_mode", MT_LOOKAHEAD_NONE)
        if mode == MT_LOOKAHEAD_TAF:
            n_cont = max(1, int(getattr(config, "taf_num_continuations", 10)))
            # One batched continuation call (n samples) plus n translation prompts.
            return n_cont + 1
        if mode == MT_LOOKAHEAD_SSBD:
            return max(1, int(getattr(config, "ssbd_max_num_seqs", 1)))
        return 1

    @staticmethod
    def _build_vllm_llm_kwargs(config: SimpleNamespace, llm_model_name: str) -> dict:
        """vLLM EngineArgs for Qwen3.5 text-only MT (see vLLM Qwen3.5 recipe)."""
        llm_quantization, llm_quantization_config = _resolve_llm_quantization(config)
        kwargs = {
            "model": llm_model_name,
            "trust_remote_code": True,
            "language_model_only": getattr(config, "llm_language_model_only", True),
            "gpu_memory_utilization": getattr(config, "llm_gpu_memory_utilization", 0.75),
            "tensor_parallel_size": getattr(config, "llm_tensor_parallel_size", 1),
            "max_num_seqs": CascadeSpeechProcessor._resolve_llm_max_num_seqs(config),
            "max_model_len": getattr(config, "llm_max_model_len", 8192),
            "enable_prefix_caching": True,
        }
        enforce_eager = getattr(config, "llm_enforce_eager", False)
        if llm_quantization == "bitsandbytes":
            # Required by vLLM for bitsandbytes (CUDA graphs not supported yet).
            enforce_eager = True
        if enforce_eager:
            kwargs["enforce_eager"] = True
        reasoning_parser = getattr(config, "llm_reasoning_parser", None)
        if reasoning_parser is None and _is_qwen35_model(llm_model_name):
            reasoning_parser = "qwen3"
        if reasoning_parser:
            kwargs["reasoning_parser"] = reasoning_parser
        max_cudagraph = getattr(config, "llm_max_cudagraph_capture_size", None)
        if max_cudagraph is not None:
            kwargs["max_cudagraph_capture_size"] = max_cudagraph
        if llm_quantization is not None:
            kwargs["quantization"] = llm_quantization
            if (
                llm_quantization_config is not None
                and llm_quantization in _LLM_ONLINE_QUANTIZATION
            ):
                kwargs["quantization_config"] = llm_quantization_config
        llm_dtype = _resolve_llm_dtype(config, llm_model_name, llm_quantization)
        if llm_dtype is not None:
            kwargs["dtype"] = llm_dtype
        return kwargs

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "asr") or cls.asr is None:
            beam_size = getattr(config, "sfm_decode", 1)
            boosting_alpha = getattr(config, "sfm_boosting_tree_alpha", 1.0)
            boosting_cfg = {
                "context_score": getattr(config, "sfm_context_score", 1.0),
                "depth_scaling": getattr(config, "sfm_depth_scaling", 2.0),
            }
            decoding_cfg = {
                "strategy": "greedy_batch" if beam_size == 1 else "malsd_batch",
                "greedy": {
                    "boosting_tree": boosting_cfg,
                    "boosting_tree_alpha": boosting_alpha,
                },
                "beam": {
                    "boosting_tree": boosting_cfg,
                    "boosting_tree_alpha": boosting_alpha,
                    "beam_size": beam_size,
                },
            }

            # Simulstream controls the audio chunk size via `speech_chunk_size` (seconds).
            # To keep Nemo's internal streaming buffer consistent, default our SFM `chunk_secs`
            # to `speech_chunk_size` when `sfm_chunk_secs` isn't explicitly provided.
            speech_chunk_size = getattr(config, "speech_chunk_size", None)
            chunk_secs_default = speech_chunk_size if speech_chunk_size is not None else 1.0

            # Accept both `sfm_*` keys (this agent's convention) and legacy/non-prefixed keys
            # to reduce chances of misconfiguration.
            left_context_secs = getattr(
                config,
                "sfm_left_context_secs",
                getattr(config, "left_context_secs", 20.0),
            )
            right_context_secs = getattr(
                config,
                "sfm_right_context_secs",
                getattr(config, "right_context_secs", 0.0),
            )
            chunk_secs = getattr(config, "sfm_chunk_secs", chunk_secs_default)

            cfg_args = SimpleNamespace(
                model_path=getattr(config, "sfm_model_path", None),
                pretrained_name=getattr(config, "sfm_pretrained_name", "nvidia/parakeet-tdt-0.6b-v3"),
                manifest_path=getattr(config, "sfm_manifest_path", "vp.jsonl"),
                chunk_secs=chunk_secs,
                left_context_secs=left_context_secs,
                right_context_secs=right_context_secs,
                max_empty_chunks=getattr(config, "sfm_max_empty_chunks", 0),
                policy=getattr(config, "sfm_policy", "LACP"),
                lacp_threshold=getattr(config, "sfm_lacp_threshold", 2.0),
                K=getattr(config, "sfm_K", 2),
                N=getattr(config, "sfm_N", 5),
                word_level=getattr(config, "sfm_word_level", False),
                slcp_semantic_threshold=getattr(config, "sfm_slcp_semantic_threshold", 0.65),
                slcp_max_gap=getattr(config, "sfm_slcp_max_gap", 3),
                slcp_use_spacy=getattr(config, "sfm_slcp_use_spacy", False),
                device=getattr(config, "sfm_device", "cuda"),
                compute_dtype=getattr(config, "sfm_compute_dtype", "bfloat16"),
                emit_incomplete=getattr(config, "sfm_emit_incomplete", False),
                rnnt_decoding=decoding_cfg,
            )
            asr_cfg = OmegaConf.create(vars(cfg_args))
            with open_dict(asr_cfg):
                asr_cfg.cuda = 0 if cfg_args.device == "cuda" else -1
                asr_cfg.allow_mps = True if cfg_args.device == "mps" else False
            cls.asr_cfg = asr_cfg
            cls.asr = StreamingParakeet(asr_cfg, mbr=getattr(config, "sfm_mbr", False))
            logger.info(
                "ASR streaming context samples (left/chunk/right): %s/%s/%s",
                cls.asr.context_samples.left,
                cls.asr.context_samples.chunk,
                cls.asr.context_samples.right,
            )

        llm_model_name = getattr(config, "llm_model_name", DEFAULT_LLM_MODEL_NAME)
        llm_base_url = getattr(config, "llm_base_url", None)
        if llm_base_url is not None:
            if not hasattr(cls, "llm_client") or cls.llm_client is None:
                cls.llm_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
                from transformers import AutoTokenizer

                cls.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            cls.llm = None
        else:
            cls.llm_client = None
            if not hasattr(cls, "llm") or cls.llm is None:
                cls.llm = LLM(**cls._build_vllm_llm_kwargs(config, llm_model_name))
                #from transformers import AutoTokenizer

                cls.tokenizer = cls.llm.get_tokenizer()
                #cls.tokenizer = AutoTokenzier.from_pretrained(llm_model_name)

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.load_model(config)

        self.source_lang = getattr(config, "source_lang", "English")
        self.target_lang = getattr(config, "target_lang", "German")
        self.target_sep = "" if self.target_lang in ["Chinese", "Japanese"] else " "
        self._spaced_target = self.target_lang not in ["Chinese", "Japanese"]
        self.latency_unit = getattr(config, "latency_unit", "word")

        self.min_start_seconds = getattr(config, "min_start_seconds", 1.0)
        self._temperature = getattr(config, "temperature", 0.7)
        self._top_p = getattr(config, "top_p", 0.9)
        self._top_k = getattr(config, "top_k", 20)
        self._max_tokens = getattr(config, "max_new_tokens", 256)
        self._repetition_penalty = getattr(config, "repetition_penalty", 1.05)
        temp_fall = getattr(config, "temp_fall", False)
        if temp_fall is True:
            self._temp_fall = [0.2, 0.4, 0.6, 0.8, 1.0]
        elif temp_fall:
            self._temp_fall = list(temp_fall)
        else:
            self._temp_fall = None
        self._compression_ratio_threshold = getattr(config, "compression_ratio_threshold", 2.4)
        self._iters_till_fallback = getattr(config, "iters_till_fallback", 4)
        self._fallback_word_rollback = getattr(config, "fallback_word_rollback", 2)
        self._llm_model_name = getattr(config, "llm_model_name", DEFAULT_LLM_MODEL_NAME)
        self._llm_max_model_len = getattr(config, "llm_max_model_len", 8192)
        # Qwen3.5 defaults to thinking mode; disable for direct translation output.
        self._llm_enable_thinking = getattr(config, "llm_enable_thinking", False)

        mt_lookahead_mode = getattr(config, "mt_lookahead_mode", MT_LOOKAHEAD_NONE)
        if mt_lookahead_mode not in MT_LOOKAHEAD_MODES:
            raise ValueError(
                f"Unsupported mt_lookahead_mode={mt_lookahead_mode!r}; "
                f"use one of {sorted(MT_LOOKAHEAD_MODES)}"
            )
        if mt_lookahead_mode == MT_LOOKAHEAD_EAC:
            logger.warning(
                "mt_lookahead_mode=eac is implemented in agent_simulstream.py (PAC); "
                "falling back to committed-ASR-only MT in agent_simulstream_new.py"
            )
            mt_lookahead_mode = MT_LOOKAHEAD_NONE
        self._mt_lookahead_mode = mt_lookahead_mode

        # TAF-style source anticipation (Ouyang et al., NAACL 2025).
        self._taf_num_continuations = max(1, int(getattr(config, "taf_num_continuations", 10)))
        self._taf_max_pred_words = max(1, int(getattr(config, "taf_max_pred_words", 6)))
        self._taf_agree_thres = float(getattr(config, "taf_agree_thres", 0.6))
        self._taf_min_start_words = max(0, int(getattr(config, "taf_min_start_words", 3)))
        self._taf_temperature = float(getattr(config, "taf_temperature", 0.7))
        self._taf_include_committed_hypothesis = bool(
            getattr(config, "taf_include_committed_hypothesis", False)
        )
        self._taf_mt_beam_width = max(1, int(getattr(config, "taf_mt_beam_width", 1)))
        self._taf_mt_beam_length_penalty = float(
            getattr(config, "taf_mt_beam_length_penalty", 1.0)
        )
        self._taf_ralcp_withhold = bool(getattr(config, "taf_ralcp_withhold", False))

        # SSBD-style self-speculative biased decoding (Zeng et al., 2026).
        self._ssbd_bias_beta = float(getattr(config, "ssbd_bias_beta", 0.2))
        self._ssbd_mask_k = max(0, int(getattr(config, "ssbd_mask_k", 0)))
        self._ssbd_logit_bias_scale = float(getattr(config, "ssbd_logit_bias_scale", 25.0))
        self._ssbd_batched_verify = bool(getattr(config, "ssbd_batched_verify", False))
        self._ssbd_raw_emission = bool(getattr(config, "ssbd_raw_emission", False))

        self.sampling_params = SamplingParams(
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
            repetition_penalty=self._repetition_penalty,
        )

        self._state = self._fresh_state(speech_id=0)

        # ---- Audio I/O sanity checks ---------------------------------------
        # SimulStream provides audio chunks at `simulstream`'s SAMPLE_RATE.
        # NeMo/Parakeet may expect a different sample rate; if so, the internal
        # streaming context bookkeeping can drift and trigger assertions.
        self._input_sample_rate = SAMPLE_RATE
        self._asr_sample_rate = self.asr.sample_rate
        self._needs_resample = self._input_sample_rate != self._asr_sample_rate

        speech_chunk_size = getattr(config, "speech_chunk_size", None)
        self._expected_input_chunk_samples = None
        if speech_chunk_size is not None:
            # Expected samples for a "full" SimulStream chunk. The final chunk
            # is often shorter; we use this to better set NeMo's last-chunk flag.
            self._expected_input_chunk_samples = int(round(float(speech_chunk_size) * self._input_sample_rate))

        self._saw_last_nonempty_chunk = False
        logger.info(
            "Audio sample rates: simulstream=%s Hz, asr=%s Hz, resample=%s",
            self._input_sample_rate,
            self._asr_sample_rate,
            self._needs_resample,
        )

    def _maybe_resample(self, waveform: np.ndarray) -> np.ndarray:
        if waveform is None or len(waveform) == 0:
            return waveform
        if not self._needs_resample:
            return waveform

        # Ensure 1D float32
        w = np.asarray(waveform).reshape(-1).astype(np.float32)

        try:
            import librosa

            return librosa.resample(w, orig_sr=self._input_sample_rate, target_sr=self._asr_sample_rate).astype(
                np.float32
            )
        except Exception as e:
            raise RuntimeError(
                "Sample-rate mismatch between SimulStream and ASR, and resampling failed. "
                f"simulstream SAMPLE_RATE={self._input_sample_rate}, asr sample_rate={self._asr_sample_rate}. "
                "Install librosa or fix the sample rates."
            ) from e

    def _fresh_state(self, speech_id: int) -> CascadeState:
        asr_buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size=1,
            context_samples=self.asr.context_samples,
            dtype=self.asr.dtype,
            device=self.asr.device,
        )
        asr_hyp_buffer = self._build_asr_hyp_buffer()
        return CascadeState(
            speech_id=speech_id,
            asr_buffer=asr_buffer,
            asr_hyp_buffer=asr_hyp_buffer,
        )

    def _build_asr_hyp_buffer(self):
        cfg = self.asr_cfg
        word_level = getattr(cfg, "word_level", False)
        if cfg.policy == "LCP":
            return LCPHypothesisBuffer(word_level=word_level, debug=False)
        if cfg.policy == "LACP":
            return LACPHypothesisBuffer(cfg.lacp_threshold, word_level=word_level, debug=False)
        if cfg.policy == "SLCP":
            return _build_slcp_buffer(cfg, word_level=word_level, debug=False)
        if cfg.policy == "WaitK":
            return WaitKHypothesisBuffer(
                cfg.K,
                features_per_second=self.asr.features_per_sec,
                subsampling_factor=self.asr.subsampling_factor,
                word_level=word_level,
                debug=False,
            )
        return HoldNHypothesisBuffer(cfg.N, word_level=word_level, debug=False)

    def _tokens_to_text(self, toks: List[str]) -> str:
        word_level = getattr(self.asr_cfg, "word_level", False)
        if word_level:
            return "".join(t.replace("▁", " ") for t in toks).strip()
        return self.asr.asr_model.tokenizer.tokens_to_text(toks)

    def _asr_step(
        self, state: CascadeState, waveform: np.ndarray, is_last_chunk: bool
    ) -> str:
        if (waveform is None or len(waveform) == 0) and not is_last_chunk:
            logger.warning(f"[ASR] Received empty waveform. Returning empty string.")
            return ""

        if waveform is not None and len(waveform) > 0:
            waveform = self._maybe_resample(waveform)
            chunk = np.asarray(waveform, dtype=np.float32)
            chunk_t = torch.tensor([chunk], device=self.asr.device)
            stride = state.asr_buffer.add_audio_batch_get_stride(
                chunk_t,
                audio_lengths=torch.tensor([len(chunk)], device=self.asr.device),
                is_last_chunk=is_last_chunk,
                is_last_chunk_batch=torch.tensor([is_last_chunk], device=self.asr.device),
            )
            state.current_offset += stride // self.asr.encoder_frame2audio_samples

            hyp = self.asr.process_chunk(state.asr_buffer, state.current_offset)
            state.asr_hyp_buffer.insert(hyp)

        max_empty_chunks = getattr(self.asr_cfg, "max_empty_chunks", 0)
        if self.asr_cfg.policy == "WaitK":
            out = state.asr_hyp_buffer.flush(last_instant=0)
        elif self.asr_cfg.policy == "LACP":
            out = state.asr_hyp_buffer.flush(forced=True)
        elif self.asr_cfg.policy == "LCP" and state.nchunks_no_output >= max_empty_chunks:
            out = state.asr_hyp_buffer.flush(forced=True)
        else:
            out = state.asr_hyp_buffer.flush()

        if is_last_chunk:
            out.extend(state.asr_hyp_buffer.complete())

        if max_empty_chunks:
            if not out:
                state.nchunks_no_output += 1
            else:
                state.nchunks_no_output = 0

        if not out:
            logger.info(f"[ASR] {self.asr_cfg.policy} policy generated no transcription. Emitting empty string")
            return ""

        res = self._tokens_to_text([t for _, _, t in out])
        logger.info(f"[ASR] Emitting committed '{res}'")
        return res

    def _count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def _apply_chat_template(self, messages: list[dict]) -> str:
        # transformers>=5 uses positional `conversation`; older builds also accepted `messages=`.
        common = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                **common,
                enable_thinking=self._llm_enable_thinking,
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(
                    messages=messages,
                    **common,
                    enable_thinking=self._llm_enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **common)

    def _build_llm_prompt(self, asr_segment: str, prev_translation: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional simultaneous speech translator. "
                    f"Translate from {self.source_lang} into {self.target_lang}. "
                    "Preserve named entities exactly as in the source text. "
                    "Output only the translation with no explanation, preamble, or reasoning."
                ),
            },
            {"role": "user", "content": asr_segment},
        ]
        prompt = self._apply_chat_template(messages)
        return prompt + prev_translation

    def _normalize_translation_text(self, text: str) -> str:
        if not text:
            return text
        text = strip_ellipsis(text)
        #if not self._spaced_target:
        #    return text
        return text
        #return insert_space_after_punctuation(text)

    def _trim_translation_increment(self, increment: str) -> str:
        """Drop trailing whitespace only; keep leading spaces for word-level emission."""
        if not increment or not self._spaced_target:
            return increment
        return increment.rstrip()

    def _sanitize_llm_output(self, text: str) -> str:
        """Drop Qwen thinking/reasoning prefixes if the model emits them anyway."""
        if not text:
            return ""
        cleaned = strip_ellipsis(text)
        think_close = "</" + "think" + ">"
        if think_close in cleaned:
            cleaned = cleaned.split(think_close, 1)[-1]
        cleaned = re.sub(
            r"(?is)^\s*thinking\s*process\s*:\s*.*?(?=\n\n|\Z)",
            "",
            cleaned,
            count=1,
        )
        return self._normalize_translation_text(cleaned.lstrip())

    def _truncate_text_from_left(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        decoded = self.tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=True)
        return self._normalize_translation_text(decoded)

    def _fit_llm_prompt(self, asr_segment: str, prev_translation: str) -> tuple[str, str, str]:
        """
        Build an LLM prompt that fits within the configured context window.

        Long ACL segments can exceed ``llm_max_model_len`` once ASR and the committed
        translation prefix grow; drop older ASR first, then older translation prefix.
        """
        reserve_tokens = self._max_tokens + 32
        max_input_tokens = max(64, self._llm_max_model_len - reserve_tokens)

        asr_words = asr_segment.split()
        prev = self._normalize_translation_text(prev_translation)
        prompt = self._build_llm_prompt(asr_segment, prev)
        prompt_tokens = self._count_prompt_tokens(prompt)

        if prompt_tokens > max_input_tokens and len(asr_words) > 1:
            lo, hi = 0, len(asr_words)
            best_asr = asr_segment
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = " ".join(asr_words[mid:])
                candidate_prompt = self._build_llm_prompt(candidate, prev)
                if self._count_prompt_tokens(candidate_prompt) <= max_input_tokens:
                    best_asr = candidate
                    hi = mid
                else:
                    lo = mid + 1
            asr_segment = best_asr
            prompt = self._build_llm_prompt(asr_segment, prev)
            prompt_tokens = self._count_prompt_tokens(prompt)
            if lo > 0:
                logger.warning(
                    "Truncated ASR prompt from %d to %d words to fit %d input tokens",
                    len(asr_words),
                    len(asr_segment.split()),
                    max_input_tokens,
                )

        if prompt_tokens > max_input_tokens and prev:
            template = self._build_llm_prompt(asr_segment, "")
            template_tokens = self._count_prompt_tokens(template)
            prev_budget = max(0, max_input_tokens - template_tokens)
            prev = self._normalize_translation_text(
                self._truncate_text_from_left(prev, prev_budget)
            )
            prompt = self._build_llm_prompt(asr_segment, prev)
            prompt_tokens = self._count_prompt_tokens(prompt)
            logger.warning(
                "Truncated committed translation prefix to %d tokens to fit context window",
                self._count_prompt_tokens(prev),
            )

        if prompt_tokens > max_input_tokens:
            prompt = self._truncate_text_from_left(prompt, max_input_tokens)
            prev = ""
            logger.warning(
                "Prompt still exceeded context after truncation; dropped translation prefix"
            )

        return prompt, asr_segment, prev

    @staticmethod
    def _compression_ratio(text: str) -> float:
        """zlib ratio; values above ~2.4 often indicate repetitive / collapsed generation."""
        if not text:
            return 0.0
        text_bytes = text.encode("utf-8")
        compressed = zlib.compress(text_bytes)
        return len(text_bytes) / len(compressed)

    def _rollback_translation(self, state: CascadeState, n_units: int) -> None:
        prev = state.prev_translation
        if not prev:
            return
        if self.target_lang in ["Chinese", "Japanese"]:
            state.prev_translation = prev[:-n_units] if len(prev) >= n_units else ""
        else:
            words = prev.split()
            if len(words) >= n_units:
                state.prev_translation = self._normalize_translation_text(
                    self.target_sep.join(words[:-n_units])
                )
            else:
                state.prev_translation = ""
        state.translation_hypotheses = [state.prev_translation]
        if self._ssbd_raw_emission and self._mt_lookahead_mode == MT_LOOKAHEAD_SSBD:
            state.displayed_hypothesis = state.prev_translation
            state.full_mt_hypothesis = state.prev_translation
        logger.warning(
            "[BAD STATE] Rolled back last %d translation unit(s); committed prefix is now %r",
            n_units,
            state.prev_translation,
        )

    def _reset_translation_state(self, state: CascadeState) -> None:
        state.prev_translation = ""
        state.translation_hypotheses = [""]
        state.full_mt_hypothesis = ""
        state.consecutive_empty_mt = 0
        state.ssbd_draft_accepted_tokens = 0
        state.ssbd_draft_total_tokens = 0
        state.ssbd_verify_steps = 0
        state.ssbd_verify_steps_zero_accept = 0
        state.ssbd_cumulative_draft_tokens = 0
        state.ssbd_cumulative_accepted_tokens = 0
        state.displayed_hypothesis = ""
        state.pending_new_tokens = []
        state.pending_deleted_tokens = []

    def _committed_translation_prefix(self, state: CascadeState) -> str:
        if self._ssbd_raw_emission and self._mt_lookahead_mode == MT_LOOKAHEAD_SSBD:
            return self._normalize_translation_text(state.displayed_hypothesis)
        return self._normalize_translation_text(state.prev_translation)

    def _compute_retranslation_token_delta(
        self, old_text: str, new_text: str
    ) -> tuple[List[str], List[str]]:
        old_tokens = self._text_to_tokens(old_text)
        new_tokens = self._text_to_tokens(new_text)
        prefix_len = 0
        for old_tok, new_tok in zip(old_tokens, new_tokens):
            if old_tok == new_tok:
                prefix_len += 1
            else:
                break
        deleted = old_tokens[prefix_len:]
        added = new_tokens[prefix_len:]
        return added, deleted

    def _ssbd_verification_logit_bias(self) -> float:
        """Map paper bias ``beta`` in [0, 1] to an additive vLLM logit boost."""
        beta = self._ssbd_bias_beta
        if beta <= 0.0:
            return 0.0
        if beta >= 1.0:
            return 100.0
        return self._ssbd_logit_bias_scale * beta

    def _ssbd_draft_suffix(self, state: CascadeState, prev_prefix: str) -> str:
        full_prev = self._normalize_translation_text(state.full_mt_hypothesis)
        prev = self._normalize_translation_text(prev_prefix)
        if not full_prev:
            return ""
        if prev and full_prev.startswith(prev):
            return full_prev[len(prev) :]
        if prev and prev in full_prev:
            idx = full_prev.index(prev) + len(prev)
            return full_prev[idx:]
        return full_prev

    def _ssbd_draft_token_ids(self, prompt: str, draft_suffix: str) -> List[int]:
        """
        Tokenize ``draft_suffix`` in the context of ``prompt``.

        Standalone ``encode(draft_suffix)`` breaks BPE boundaries when ``prompt``
        already ends with the committed translation prefix.
        """
        if not draft_suffix:
            return []
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        combined_ids = self.tokenizer.encode(
            prompt + draft_suffix,
            add_special_tokens=False,
        )
        if len(combined_ids) <= len(prompt_ids):
            return []
        if combined_ids[: len(prompt_ids)] != prompt_ids:
            logger.warning(
                "[SSBD] Prompt is not a token prefix of prompt+draft; "
                "falling back to contextual re-encode"
            )
        return combined_ids[len(prompt_ids) :]

    def _ssbd_record_verify_step(self, state: CascadeState) -> None:
        state.ssbd_verify_steps += 1
        state.ssbd_cumulative_draft_tokens += state.ssbd_draft_total_tokens
        state.ssbd_cumulative_accepted_tokens += state.ssbd_draft_accepted_tokens
        if state.ssbd_draft_accepted_tokens == 0 and state.ssbd_draft_total_tokens > 0:
            state.ssbd_verify_steps_zero_accept += 1

    def _ssbd_log_step_acceptance(self, state: CascadeState) -> None:
        total = state.ssbd_draft_total_tokens
        accepted = state.ssbd_draft_accepted_tokens
        if total <= 0:
            return
        rate = 100.0 * accepted / total
        mode = "batched" if self._ssbd_batched_verify else "tokenwise"
        beta = 0.0 if self._ssbd_batched_verify else self._ssbd_bias_beta
        logger.info(
            "[SSBD] Draft acceptance %d/%d tokens (%.1f%%, beta=%.2f, %s)",
            accepted,
            total,
            rate,
            beta,
            mode,
        )

    def _ssbd_log_utterance_summary(self, state: CascadeState) -> None:
        if state.ssbd_verify_steps <= 0:
            return
        draft_total = state.ssbd_cumulative_draft_tokens
        accepted_total = state.ssbd_cumulative_accepted_tokens
        token_rate = 100.0 * accepted_total / draft_total if draft_total else 0.0
        zero_steps = state.ssbd_verify_steps_zero_accept
        step_rate = 100.0 * zero_steps / state.ssbd_verify_steps
        logger.info(
            "[SSBD] Utterance %d summary: verify_steps=%d, "
            "token_acceptance=%d/%d (%.1f%%), zero_accept_steps=%d (%.1f%%)",
            state.speech_id,
            state.ssbd_verify_steps,
            accepted_total,
            draft_total,
            token_rate,
            zero_steps,
            step_rate,
        )

    def _ssbd_apply_display_mask(self, increment: str) -> str:
        """Display-only mask-k: hide the last k emitted words from the user."""
        increment = self._trim_translation_increment(
            self._normalize_translation_text(increment or "")
        )
        if self._ssbd_mask_k <= 0 or not increment.strip():
            return increment
        if self.latency_unit in ["word", "spm"]:
            words = [tok for tok in increment.strip().split() if tok]
            if len(words) <= self._ssbd_mask_k:
                return ""
            masked = self.target_sep.join(words[: -self._ssbd_mask_k])
            if (
                self._spaced_target
                and increment.startswith(" ")
                and masked
                and not masked.startswith(" ")
            ):
                masked = " " + masked
            return masked
        chars = list(increment)
        if len(chars) <= self._ssbd_mask_k:
            return ""
        return "".join(chars[: -self._ssbd_mask_k])

    def _llm_generate_on_prompt(
        self,
        state: CascadeState,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logit_bias: Optional[dict[int, float]] = None,
    ) -> tuple[str, List[int]]:
        self._record_llm_calls(state, 1)
        prompt_tokens = self._count_prompt_tokens(prompt)
        if max_tokens is None:
            max_tokens = min(
                self._max_tokens, max(1, self._llm_max_model_len - prompt_tokens - 1)
            )
        temp = self._temperature if temperature is None else temperature

        if self.llm_client is not None:
            extra_body = {
                "repetition_penalty": self._repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": self._llm_enable_thinking},
            }
            if logit_bias:
                extra_body["logit_bias"] = logit_bias
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self._top_p,
                extra_body=extra_body,
            )
            text = self._sanitize_llm_output(response.choices[0].text)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return text, token_ids

        sampling_params = SamplingParams(
            temperature=temp,
            top_p=self._top_p,
            top_k=self._top_k,
            max_tokens=max_tokens,
            repetition_penalty=self._repetition_penalty,
            logit_bias=logit_bias,
        )
        llm_outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        output = llm_outputs[0].outputs[0]
        text = self._sanitize_llm_output(output.text)
        token_ids = list(output.token_ids)
        return text, token_ids

    def _ssbd_count_draft_acceptance(
        self,
        generated_ids: List[int],
        draft_token_ids: List[int],
    ) -> int:
        compare_len = min(len(generated_ids), len(draft_token_ids))
        for i in range(compare_len):
            if generated_ids[i] != draft_token_ids[i]:
                logger.debug(
                    "[SSBD] Draft mismatch at token %d: draft=%s generated=%s "
                    "(draft_tok=%r gen_tok=%r)",
                    i,
                    draft_token_ids[i],
                    generated_ids[i],
                    self.tokenizer.decode(
                        [draft_token_ids[i]], skip_special_tokens=False
                    ),
                    self.tokenizer.decode(
                        [generated_ids[i]], skip_special_tokens=False
                    ),
                )
                return i
        return compare_len

    def _ssbd_generate_from_draft_batched(
        self,
        state: CascadeState,
        prompt: str,
        draft_suffix: str,
    ) -> str:
        """
        Option A: one greedy generate verifies the draft prefix (beta=0) and
        continues in the same KV-cached pass instead of per-token prefills.
        """
        draft_token_ids = self._ssbd_draft_token_ids(prompt, draft_suffix)
        state.ssbd_draft_total_tokens = len(draft_token_ids)
        state.ssbd_draft_accepted_tokens = 0

        if not draft_token_ids:
            text, _ = self._llm_generate_on_prompt(state, prompt)
            return text

        if self._ssbd_bias_beta > 0.0:
            logger.debug(
                "[SSBD] batched verify ignores ssbd_bias_beta=%.2f (uses beta=0)",
                self._ssbd_bias_beta,
            )

        prompt_tokens = self._count_prompt_tokens(prompt)
        remaining = min(
            self._max_tokens,
            max(1, self._llm_max_model_len - prompt_tokens - 1),
        )
        max_tokens = min(
            len(draft_token_ids) + remaining,
            max(1, self._llm_max_model_len - prompt_tokens - 1),
        )

        text, generated_ids = self._llm_generate_on_prompt(
            state,
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        state.ssbd_draft_accepted_tokens = self._ssbd_count_draft_acceptance(
            generated_ids, draft_token_ids
        )

        self._ssbd_record_verify_step(state)
        self._ssbd_log_step_acceptance(state)
        return self._normalize_translation_text(text)

    def _ssbd_generate_from_draft_tokenwise(
        self,
        state: CascadeState,
        prompt: str,
        draft_suffix: str,
    ) -> str:
        draft_token_ids = self._ssbd_draft_token_ids(prompt, draft_suffix)
        state.ssbd_draft_total_tokens = len(draft_token_ids)
        state.ssbd_draft_accepted_tokens = 0

        if not draft_token_ids:
            text, _ = self._llm_generate_on_prompt(state, prompt)
            return text

        working_prompt = prompt
        accepted_parts: List[str] = []
        logit_boost = self._ssbd_verification_logit_bias()

        for draft_id in draft_token_ids:
            piece, generated_ids = self._llm_generate_on_prompt(
                state,
                working_prompt,
                temperature=0.0,
                max_tokens=1,
                logit_bias={draft_id: logit_boost},
            )
            if not generated_ids:
                break
            generated_id = generated_ids[0]
            if generated_id != draft_id:
                logger.debug(
                    "[SSBD] Draft mismatch at token %d: draft=%s generated=%s "
                    "(draft_tok=%r gen_tok=%r)",
                    state.ssbd_draft_accepted_tokens,
                    draft_id,
                    generated_id,
                    self.tokenizer.decode([draft_id], skip_special_tokens=False),
                    self.tokenizer.decode([generated_id], skip_special_tokens=False),
                )
                break
            accepted_parts.append(piece)
            working_prompt += piece
            state.ssbd_draft_accepted_tokens += 1

        self._ssbd_record_verify_step(state)
        self._ssbd_log_step_acceptance(state)

        prompt_tokens = self._count_prompt_tokens(working_prompt)
        remaining = min(
            self._max_tokens,
            max(1, self._llm_max_model_len - prompt_tokens - 1),
        )
        suffix, _ = self._llm_generate_on_prompt(
            state,
            working_prompt,
            max_tokens=remaining,
        )
        accepted = "".join(accepted_parts)
        return self._normalize_translation_text(f"{accepted}{suffix}")

    def _ssbd_generate_from_draft(
        self,
        state: CascadeState,
        prompt: str,
        draft_suffix: str,
    ) -> str:
        if self._ssbd_batched_verify:
            return self._ssbd_generate_from_draft_batched(state, prompt, draft_suffix)
        return self._ssbd_generate_from_draft_tokenwise(state, prompt, draft_suffix)

    def _ssbd_generate_with_fallback(
        self,
        state: CascadeState,
        asr_text: str,
        prev_prefix: str,
    ) -> str:
        prompt, asr_text, prev_prefix = self._fit_llm_prompt(asr_text, prev_prefix)
        if prev_prefix != state.prev_translation:
            state.prev_translation = prev_prefix
            state.translation_hypotheses = [prev_prefix]

        draft_suffix = self._ssbd_draft_suffix(state, prev_prefix)
        if draft_suffix.strip():
            hypothesis = self._ssbd_generate_from_draft(state, prompt, draft_suffix)
        else:
            hypothesis, _ = self._llm_generate_on_prompt(state, prompt)

        if not hypothesis.strip():
            state.consecutive_empty_mt += 1
            if (
                self._temp_fall
                and state.consecutive_empty_mt >= self._iters_till_fallback
            ):
                self._rollback_translation(state, self._fallback_word_rollback)
                state.consecutive_empty_mt = 0
                prompt, _, prev_prefix = self._fit_llm_prompt(
                    asr_text, state.prev_translation
                )
                hypothesis, _ = self._llm_generate_on_prompt(state, prompt)
            return hypothesis

        state.consecutive_empty_mt = 0
        if not self._temp_fall:
            return hypothesis

        cs = self._compression_ratio(hypothesis)
        if cs <= self._compression_ratio_threshold:
            return hypothesis

        for temp in self._temp_fall:
            candidate, _ = self._llm_generate_on_prompt(
                state, prompt, temperature=temp
            )
            if (
                candidate.strip()
                and self._compression_ratio(candidate) < self._compression_ratio_threshold
            ):
                return candidate
        self._reset_translation_state(state)
        return ""

    def _finalize_ssbd_raw_emission(
        self,
        state: CascadeState,
        full_hypothesis: str,
    ) -> str:
        full_hypothesis = self._normalize_translation_text(full_hypothesis)
        state.full_mt_hypothesis = full_hypothesis
        added, deleted = self._compute_retranslation_token_delta(
            state.displayed_hypothesis,
            full_hypothesis,
        )
        state.displayed_hypothesis = full_hypothesis
        state.prev_translation = full_hypothesis
        state.pending_new_tokens = added
        state.pending_deleted_tokens = deleted
        logger.info(
            "[SSBD raw] deleted=%r added=%r (displayed_len=%d)",
            deleted,
            added,
            len(self._text_to_tokens(full_hypothesis)),
        )
        return ""

    def _finalize_ssbd_translation_step(
        self,
        state: CascadeState,
        prev_prefix: str,
        full_hypothesis: str,
        force_final: bool,
    ) -> str:
        if self._ssbd_raw_emission:
            return self._finalize_ssbd_raw_emission(state, full_hypothesis)

        state.full_mt_hypothesis = full_hypothesis

        if force_final:
            increment = full_hypothesis[len(prev_prefix) :]
            state.prev_translation = full_hypothesis
            return self._ssbd_apply_display_mask(increment)

        state.translation_hypotheses.append(full_hypothesis)
        stable = self._normalize_translation_text(
            longest_common_prefix(
                state.translation_hypotheses[-2],
                state.translation_hypotheses[-1],
            )
        )
        increment = stable[len(prev_prefix) :]
        state.prev_translation = stable
        masked_increment = self._ssbd_apply_display_mask(increment)
        logger.info(
            "[SSBD] Stable increment=%r display=%r (mask_k=%d)",
            self._trim_translation_increment(increment),
            self._trim_translation_increment(masked_increment),
            self._ssbd_mask_k,
        )
        return masked_increment

    def _record_llm_calls(self, state: CascadeState, n_calls: int = 1) -> None:
        state.mt_llm_calls_step += n_calls
        state.mt_llm_calls_utterance += n_calls

    def _build_source_continuation_prompt(self, committed_asr: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You continue incomplete {self.source_lang} text naturally. "
                    f"Output only the continuation (at most {self._taf_max_pred_words} words) "
                    "with no explanation, preamble, or reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Continue the following {self.source_lang} text with up to "
                    f"{self._taf_max_pred_words} words:\n\n{committed_asr}"
                ),
            },
        ]
        return self._apply_chat_template(messages)

    @staticmethod
    def _clean_source_continuation(raw: str, committed_asr: str, max_words: int) -> str:
        text = strip_ellipsis(raw or "").strip()
        if not text:
            return ""
        committed_words = committed_asr.split()
        cont_words = text.split()
        if committed_words and cont_words[: len(committed_words)] == committed_words:
            cont_words = cont_words[len(committed_words) :]
        return _truncate_to_max_words(" ".join(cont_words), max_words)

    def _llm_generate_raw_batch(
        self,
        prompts: Sequence[str],
        *,
        temperature: Optional[float] = None,
        n_per_prompt: int = 1,
    ) -> List[str]:
        if not prompts:
            return []

        temp = self._temperature if temperature is None else temperature
        max_tokens_list: List[int] = []
        for prompt in prompts:
            prompt_tokens = self._count_prompt_tokens(prompt)
            max_tokens_list.append(
                min(self._max_tokens, max(1, self._llm_max_model_len - prompt_tokens - 1))
            )
        batch_max_tokens = max(max_tokens_list)

        if self.llm_client is not None:
            outputs: List[str] = []
            for prompt in prompts:
                response = self.llm_client.completions.create(
                    model=self._llm_model_name,
                    prompt=prompt,
                    max_tokens=batch_max_tokens,
                    temperature=temp,
                    top_p=self._top_p,
                    n=n_per_prompt,
                    extra_body={
                        "repetition_penalty": self._repetition_penalty,
                        "chat_template_kwargs": {"enable_thinking": self._llm_enable_thinking},
                    },
                )
                for choice in response.choices:
                    outputs.append(self._sanitize_llm_output(choice.text))
            return outputs

        sampling_params = SamplingParams(
            temperature=temp,
            top_p=self._top_p,
            top_k=self._top_k,
            max_tokens=batch_max_tokens,
            repetition_penalty=self._repetition_penalty,
            n=n_per_prompt,
        )
        llm_outputs = self.llm.generate(
            list(prompts),
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        texts: List[str] = []
        for request_output in llm_outputs:
            for completion in request_output.outputs:
                texts.append(self._sanitize_llm_output(completion.text))
        return texts

    def _llm_beam_search_batch(
        self,
        state: CascadeState,
        prompts: Sequence[str],
    ) -> List[str]:
        """Beam-search MT for each prompt; return all beam increments (flattened)."""
        if not prompts:
            return []

        beam_width = self._taf_mt_beam_width
        max_tokens_list: List[int] = []
        for prompt in prompts:
            prompt_tokens = self._count_prompt_tokens(prompt)
            max_tokens_list.append(
                min(self._max_tokens, max(1, self._llm_max_model_len - prompt_tokens - 1))
            )
        batch_max_tokens = max(max_tokens_list)

        if self.llm_client is not None:
            outputs: List[str] = []
            for prompt in prompts:
                response = self.llm_client.completions.create(
                    model=self._llm_model_name,
                    prompt=prompt,
                    max_tokens=batch_max_tokens,
                    temperature=0.0,
                    n=beam_width,
                    extra_body={
                        "use_beam_search": True,
                        "best_of": beam_width,
                        "length_penalty": self._taf_mt_beam_length_penalty,
                        "repetition_penalty": self._repetition_penalty,
                        "chat_template_kwargs": {"enable_thinking": self._llm_enable_thinking},
                    },
                )
                for choice in response.choices:
                    outputs.append(self._sanitize_llm_output(choice.text))
            self._record_llm_calls(state, len(prompts))
            return outputs

        from vllm.sampling_params import BeamSearchParams

        beam_params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=batch_max_tokens,
            length_penalty=self._taf_mt_beam_length_penalty,
        )
        llm_outputs = self.llm.beam_search(list(prompts), beam_params)
        self._record_llm_calls(state, len(prompts))

        texts: List[str] = []
        for request_output in llm_outputs:
            for sequence in request_output.sequences:
                texts.append(self._sanitize_llm_output(sequence.text))
        return texts

    def _predict_source_continuations(self, state: CascadeState, committed_asr: str) -> List[str]:
        prompt = self._build_source_continuation_prompt(committed_asr)
        cont_max_tokens = max(8, self._taf_max_pred_words * 4)
        prompt_tokens = self._count_prompt_tokens(prompt)
        max_tokens = min(cont_max_tokens, max(1, self._llm_max_model_len - prompt_tokens - 1))

        temp = 0.0 if self._taf_num_continuations == 1 else self._taf_temperature
        n_per_prompt = self._taf_num_continuations

        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self._top_p,
                n=n_per_prompt,
                extra_body={
                    "repetition_penalty": self._repetition_penalty,
                    "chat_template_kwargs": {"enable_thinking": self._llm_enable_thinking},
                },
            )
            self._record_llm_calls(state, 1)
            raw_continuations = [choice.text for choice in response.choices]
        else:
            sampling_params = SamplingParams(
                temperature=temp,
                top_p=self._top_p,
                top_k=self._top_k,
                max_tokens=max_tokens,
                repetition_penalty=self._repetition_penalty,
                n=n_per_prompt,
            )
            llm_outputs = self.llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            self._record_llm_calls(state, 1)
            raw_continuations = [out.text for out in llm_outputs[0].outputs]

        continuations: List[str] = []
        seen: set[str] = set()
        for raw in raw_continuations:
            cleaned = self._clean_source_continuation(
                raw, committed_asr, self._taf_max_pred_words
            )
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            continuations.append(cleaned)

        if not continuations:
            continuations = [""]
        logger.info(
            "[TAF] Predicted %d source continuation(s): %r",
            len(continuations),
            continuations,
        )
        return continuations

    def _taf_translate_hypotheses(
        self,
        state: CascadeState,
        committed_asr: str,
        continuations: Sequence[str],
        prev_prefix: str,
    ) -> List[str]:
        prev_prefix = self._normalize_translation_text(prev_prefix)
        cont_list = list(continuations)
        if self._taf_include_committed_hypothesis and "" not in cont_list:
            cont_list = [""] + cont_list

        prompts: List[str] = []
        for cont in cont_list:
            source = f"{committed_asr} {cont}".strip() if cont else committed_asr
            prompt, _, _ = self._fit_llm_prompt(source, prev_prefix)
            prompts.append(prompt)

        if self._taf_mt_beam_width > 1:
            increments = self._llm_beam_search_batch(state, prompts)
        else:
            increments = self._llm_generate_raw_batch(prompts, temperature=self._temperature)
            self._record_llm_calls(state, len(prompts))

        full_hypotheses: List[str] = []
        for increment in increments:
            full_hyp = self._normalize_translation_text(
                f"{prev_prefix.strip()} {increment}".strip()
            )
            full_hypotheses.append(full_hyp)
        logger.info(
            "[TAF] MT candidates: %d prompt(s), %d hypothesis(es), beam=%d, committed=%s",
            len(prompts),
            len(full_hypotheses),
            self._taf_mt_beam_width,
            self._taf_include_committed_hypothesis,
        )
        return full_hypotheses

    def _taf_voted_increment(
        self,
        prev_prefix: str,
        hypotheses: Sequence[str],
    ) -> tuple[str, str]:
        """Return (voted_full_hypothesis, agreed_increment_after_prev_prefix)."""
        prev_prefix = self._normalize_translation_text(prev_prefix)
        voted_full = self._normalize_translation_text(
            majority_vote_prefix(hypotheses, self._taf_agree_thres)
        )
        if voted_full.startswith(prev_prefix):
            increment = voted_full[len(prev_prefix) :].strip()
        elif prev_prefix.startswith(voted_full):
            increment = ""
        else:
            increment = voted_full.strip()
        return voted_full, increment

    def _translate_with_taf_lookahead(
        self,
        state: CascadeState,
        committed_asr: str,
        prev_prefix: str,
        force_final: bool,
    ) -> str:
        committed_words = committed_asr.split()
        if len(committed_words) < self._taf_min_start_words:
            hypothesis = self._llm_generate_with_fallback(state, committed_asr, prev_prefix)
            prev_prefix = self._normalize_translation_text(state.prev_translation)
            full_hypothesis = self._normalize_translation_text(
                f"{prev_prefix.strip()} {hypothesis}".strip()
            )
            return self._finalize_translation_step(
                state, prev_prefix, full_hypothesis, force_final
            )

        continuations = self._predict_source_continuations(state, committed_asr)
        hypotheses = self._taf_translate_hypotheses(
            state, committed_asr, continuations, prev_prefix
        )
        voted_full, voted_increment = self._taf_voted_increment(prev_prefix, hypotheses)

        if self._taf_ralcp_withhold and not force_final:
            if not voted_increment:
                logger.info(
                    "[TAF] RALCP withhold: no agreed increment (%d hypotheses, thres=%.2f)",
                    len(hypotheses),
                    self._taf_agree_thres,
                )
                return ""

        if not voted_full:
            voted_full = self._normalize_translation_text(
                max(hypotheses, key=len) if hypotheses else prev_prefix
            )

        logger.info(
            "[TAF] Majority vote (%d hypotheses, thres=%.2f): %r",
            len(hypotheses),
            self._taf_agree_thres,
            voted_full[len(prev_prefix) :].strip(),
        )
        return self._finalize_translation_step(
            state, prev_prefix, voted_full, force_final
        )

    def _translate_with_ssbd_lookahead(
        self,
        state: CascadeState,
        committed_asr: str,
        prev_prefix: str,
        force_final: bool,
    ) -> str:
        if self.llm_client is not None and not self._ssbd_batched_verify:
            logger.warning(
                "[SSBD] Remote OpenAI-compatible backend lacks verified logit_bias "
                "support; falling back to standard re-translation."
            )
            hypothesis = self._llm_generate_with_fallback(state, committed_asr, prev_prefix)
        else:
            hypothesis = self._ssbd_generate_with_fallback(
                state, committed_asr, prev_prefix
            )

        prev_prefix = self._normalize_translation_text(state.prev_translation)
        full_hypothesis = self._normalize_translation_text(
            f"{prev_prefix.strip()} {hypothesis}".strip()
        )
        return self._finalize_ssbd_translation_step(
            state, prev_prefix, full_hypothesis, force_final
        )

    def _finalize_translation_step(
        self,
        state: CascadeState,
        prev_prefix: str,
        full_hypothesis: str,
        force_final: bool,
    ) -> str:
        state.full_mt_hypothesis = full_hypothesis

        if force_final:
            increment = full_hypothesis[len(prev_prefix) :]
            state.prev_translation = full_hypothesis
            return self._trim_translation_increment(increment)

        state.translation_hypotheses.append(full_hypothesis)
        stable = self._normalize_translation_text(
            longest_common_prefix(
                state.translation_hypotheses[-2],
                state.translation_hypotheses[-1],
            )
        )
        logger.info(
            "[LLM] Stable: %r",
            self._trim_translation_increment(stable[len(prev_prefix) :]),
        )
        increment = stable[len(prev_prefix) :]
        state.prev_translation = stable
        return self._trim_translation_increment(increment)

    def _llm_generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        text, _ = self._llm_generate_on_prompt(
            self._state, prompt, temperature=temperature
        )
        logger.info(f"[LLM] Response: '{text}'")
        return text

    def _llm_generate_with_fallback(self, state: CascadeState, asr_text: str, prev_prefix: str) -> str:
        """
        MT generation with Whisper-style fallbacks (segfreetk vllm.py):
        - empty output: after N consecutive empties, roll back committed translation and retry
        - repetition loop: retry with rising temperature until compression ratio drops
        """
        prompt, asr_text, prev_prefix = self._fit_llm_prompt(asr_text, prev_prefix)
        if prev_prefix != state.prev_translation:
            state.prev_translation = prev_prefix
            state.translation_hypotheses = [prev_prefix]

        hypothesis = self._llm_generate(prompt)

        logger.debug(f"[LLM] RAW Hypothesis: {hypothesis}")

        if not hypothesis.strip():
            state.consecutive_empty_mt += 1
            if (
                self._temp_fall
                and state.consecutive_empty_mt >= self._iters_till_fallback
            ):
                self._rollback_translation(state, self._fallback_word_rollback)
                state.consecutive_empty_mt = 0
                prompt, _, prev_prefix = self._fit_llm_prompt(asr_text, state.prev_translation)
                hypothesis = self._llm_generate(prompt)
                if not hypothesis.strip():
                    logger.warning(
                        "[BAD STATE] No MT output after empty-generation rollback; "
                        "keeping previous translation prefix"
                    )
            return hypothesis

        state.consecutive_empty_mt = 0

        if not self._temp_fall:
            return hypothesis

        cs = self._compression_ratio(hypothesis)
        if cs <= self._compression_ratio_threshold:
            return hypothesis

        logger.warning(
            "[BAD GENERATION] Detected repetition (compression_ratio=%.2f); "
            "retrying with temperature fallback. Output was %r",
            cs,
            hypothesis,
        )
        for temp in self._temp_fall:
            candidate = self._llm_generate(prompt, temperature=temp)
            candidate_cs = self._compression_ratio(candidate)
            if candidate.strip() and candidate_cs < self._compression_ratio_threshold:
                logger.warning(
                    "Temperature fallback succeeded at temperature=%s (compression_ratio=%.2f)",
                    temp,
                    candidate_cs,
                )
                return candidate

        final_cs = self._compression_ratio(hypothesis)
        logger.error(
            "[BAD STATE] Temperature fallback did not recover (compression_ratio=%.2f); "
            "resetting translation state",
            final_cs,
        )
        self._reset_translation_state(state)
        return ""

    def _translate_from_asr(self, state: CascadeState, force_final: bool) -> str:
        state.mt_llm_calls_step = 0
        asr_text = state.asr_committed_text.strip()
        if not asr_text:
            return ""

        prev_prefix = self._committed_translation_prefix(state)

        if self._mt_lookahead_mode == MT_LOOKAHEAD_TAF:
            increment = self._translate_with_taf_lookahead(
                state, asr_text, prev_prefix, force_final
            )
        elif self._mt_lookahead_mode == MT_LOOKAHEAD_SSBD:
            increment = self._translate_with_ssbd_lookahead(
                state, asr_text, prev_prefix, force_final
            )
        else:
            hypothesis = self._llm_generate_with_fallback(state, asr_text, prev_prefix)
            prev_prefix = self._normalize_translation_text(state.prev_translation)
            full_hypothesis = self._normalize_translation_text(
                f"{prev_prefix.strip()} {hypothesis}".strip()
            )
            increment = self._finalize_translation_step(
                state, prev_prefix, full_hypothesis, force_final
            )

        logger.info(
            "[MT] Lookahead=%s; LLM calls this step=%d (utterance total=%d)",
            self._mt_lookahead_mode,
            state.mt_llm_calls_step,
            state.mt_llm_calls_utterance,
        )
        return increment

    def _text_to_tokens(self, text: str) -> List[str]:
        if text == "":
            return []
        text = self._normalize_translation_text(text)
        if self.latency_unit in ["word", "spm"]:
            return [tok for tok in text.strip().split() if tok]
        if self.latency_unit == "char":
            return list(text.strip())
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def _build_incremental_output(self, stable_increment: str) -> IncrementalOutput:
        if self._ssbd_raw_emission and self._mt_lookahead_mode == MT_LOOKAHEAD_SSBD:
            new_tokens = list(self._state.pending_new_tokens)
            deleted_tokens = list(self._state.pending_deleted_tokens)
            self._state.pending_new_tokens = []
            self._state.pending_deleted_tokens = []
        else:
            stable_increment = self._trim_translation_increment(
                self._normalize_translation_text(stable_increment or "")
            )
            new_tokens = self._text_to_tokens(stable_increment)
            deleted_tokens = []

        if not new_tokens and not deleted_tokens:
            return IncrementalOutput([], "", [], "")

        new_string = self.tokens_to_string(new_tokens) if new_tokens else ""
        if (
            self.latency_unit == "word"
            and self._state.emission_started
            and new_string
            and not new_string.startswith(" ")
        ):
            new_string = " " + new_string

        deleted_string = self.tokens_to_string(deleted_tokens) if deleted_tokens else ""
        if new_tokens or deleted_tokens:
            self._state.emission_started = True

        if deleted_tokens:
            logger.info(
                "[SSBD raw] Retracting %r; emitting %r",
                deleted_string,
                new_string,
            )

        return IncrementalOutput(
            new_tokens=new_tokens,
            new_string=new_string,
            deleted_tokens=deleted_tokens,
            deleted_string=deleted_string,
        )

    @torch.inference_mode()
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        logger.info(f"================ Performing new step ================")
        if waveform is None or len(waveform) == 0:
            return IncrementalOutput([], "", [], "")

        self._state.total_samples += len(waveform)
        total_duration = self._state.total_samples / SAMPLE_RATE
        if total_duration < self.min_start_seconds and self._state.asr_committed_text == "":
            return IncrementalOutput([], "", [], "")

        # Best-effort "last chunk" detection:
        # SimulStream often sends a final shorter chunk right before end_of_stream().
        is_last_chunk = False
        if self._expected_input_chunk_samples is not None and len(waveform) < self._expected_input_chunk_samples:
            is_last_chunk = True
            self._saw_last_nonempty_chunk = True

        asr_increment = self._asr_step(
            self._state, waveform, is_last_chunk=is_last_chunk
        )
        if asr_increment:
            if self._state.asr_committed_text:
                self._state.asr_committed_text = f"{self._state.asr_committed_text} {asr_increment}".strip()
            else:
                self._state.asr_committed_text = asr_increment.strip()

        translation_increment = self._translate_from_asr(self._state, force_final=False)
        return self._build_incremental_output(translation_increment)

    @torch.inference_mode()
    def end_of_stream(self) -> IncrementalOutput:
        # If we already treated the last non-empty chunk as `is_last_chunk=True`,
        # avoid double "end" notifications to NeMo.
        asr_increment = self._asr_step(
            self._state,
            np.zeros(0, dtype=np.float32),
            is_last_chunk=not self._saw_last_nonempty_chunk,
        )
        if asr_increment:
            if self._state.asr_committed_text:
                self._state.asr_committed_text = f"{self._state.asr_committed_text} {asr_increment}".strip()
            else:
                self._state.asr_committed_text = asr_increment.strip()

        translation = self._translate_from_asr(self._state, force_final=True)
        if self._mt_lookahead_mode == MT_LOOKAHEAD_SSBD:
            self._ssbd_log_utterance_summary(self._state)
        logger.info(
            "[MT] Utterance %d finished; total LLM calls=%d",
            self._state.speech_id,
            self._state.mt_llm_calls_utterance,
        )
        current_speech_id = self._state.speech_id + 1
        self._state = self._fresh_state(speech_id=current_speech_id)
        return self._build_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        self.source_lang = language

    def set_target_language(self, language: str) -> None:
        self.target_lang = language
        self.target_sep = "" if language in ["Chinese", "Japanese"] else " "
        self._spaced_target = language not in ["Chinese", "Japanese"]

    def tokens_to_string(self, tokens: List[str]) -> str:
        if self.latency_unit in ["word", "spm"]:
            return " ".join(tokens)
        if self.latency_unit == "char":
            return "".join(tokens)
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def clear(self) -> None:
        self._state = self._fresh_state(speech_id=self._state.speech_id)
