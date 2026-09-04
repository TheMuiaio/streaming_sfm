import logging
import re
import os
import unicodedata
import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

# CUDA allocator tuning; must be set before the first `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
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
from streaming_sfm.aligners import AlignmentResult, WordAligner
from streaming_sfm.eval_progress import build_eval_progress_index, format_step_banner
from streaming_sfm.streaming_model import (
    StreamingBatchedAudioBufferWithOffset,
    StreamingParakeet,
)

try:
    from segfreetk.states.sclp import _LinguisticChecks, _slcp_stable_flags

    _MT_SLCP_AVAILABLE = True
except ImportError:
    _LinguisticChecks = None  # type: ignore[misc, assignment]
    _slcp_stable_flags = None  # type: ignore[misc, assignment]
    _MT_SLCP_AVAILABLE = False

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


def longest_common_prefix(s1: str, s2: str, *, word_level: bool = True) -> str:
    if not word_level:
        limit = min(len(s1), len(s2))
        for i in range(limit):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1[:limit]

    t1 = s1.split()
    t2 = s2.split()
    for i in range(min(len(t1), len(t2))):
        if t1[i] != t2[i]:
            return " ".join(t1[:i])
    return " ".join(t1[: min(len(t1), len(t2))])


def word_lcp_len(a: List[str], b: List[str]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def word_lacp_prefix_tokens(
    prev_tokens: List[str],
    curr_tokens: List[str],
    threshold: float,
    uncased: bool = True,
    *,
    word_level: bool = True,
) -> List[str]:
    """Longest prefix of ``curr_tokens`` stable under the LACP Levenshtein threshold."""
    joiner = " " if word_level else ""
    commit: List[str] = []
    past: List[str] = []
    present: List[str] = []
    for i in range(min(len(prev_tokens), len(curr_tokens))):
        past.append(prev_tokens[i])
        present.append(curr_tokens[i])
        past_s = joiner.join(past)
        present_s = joiner.join(present)
        if uncased:
            past_s = past_s.lower()
            present_s = present_s.lower()
        if Levenshtein.distance(past_s, present_s) > threshold:
            break
        commit.append(curr_tokens[i])
    return commit


_CHAR_BASED_TARGET_LANGS = frozenset(
    {"Chinese", "Japanese", "Mandarin", "Cantonese", "Korean"}
)


def _is_char_based_target_lang(language: str) -> bool:
    if language in _CHAR_BASED_TARGET_LANGS:
        return True
    lowered = language.lower()
    return any(
        token in lowered
        for token in ("chinese", "japanese", "mandarin", "cantonese", "korean")
    )


@dataclass
class CascadeState:
    speech_id: int = 0
    asr_committed_text: str = ""
    speculative_text: str = ""
    last_alignment: Optional[AlignmentResult] = None
    first_speculative_word_idx: Optional[int] = None
    current_speculative_tokens: List[str] = field(default_factory=list)
    last_emitted_speculative_tokens: List[str] = field(default_factory=list)
    last_emitted_tokens: List[str] = field(default_factory=list)
    cumulative_emitted_tokens: List[str] = field(default_factory=list)
    stream_speculative_suffix_len: int = 0
    prev_translation: str = ""
    translation_hypotheses: List[str] = field(default_factory=lambda: [""])
    mt_slcp_prev_hyp: List[str] = field(default_factory=list)
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
    """

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
            "max_num_seqs": 1,
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
    def _run_asr_warmup_pass(cls, asr: StreamingParakeet, *, num_chunks: int) -> None:
        """Drive ASR through ``num_chunks`` streaming forwards on dummy audio."""
        chunk_samples = asr.context_samples.chunk
        if chunk_samples <= 0:
            return

        buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size=1,
            context_samples=asr.context_samples,
            dtype=asr.dtype,
            device=asr.device,
        )
        # Low-amplitude noise avoids RNNT decoder edge cases with all-zero frames.
        chunk_audio = (
            torch.randn((1, chunk_samples), dtype=asr.dtype, device=asr.device) * 1e-3
        )
        offset = 0
        for i in range(num_chunks):
            is_last = i == num_chunks - 1
            stride = buffer.add_audio_batch_get_stride(
                chunk_audio,
                audio_lengths=torch.tensor([chunk_samples], device=asr.device),
                is_last_chunk=is_last,
                is_last_chunk_batch=torch.tensor([is_last], device=asr.device),
            )
            offset += stride // asr.encoder_frame2audio_samples
            asr.process_chunk(buffer, offset)

    @classmethod
    def _warmup_asr_model(cls, config: SimpleNamespace, *, phase: str = "pre_llm") -> None:
        """Run ASR streaming warmup on CUDA.

        ``pre_llm``: settle Parakeet allocations (~5 GiB -> ~2 GiB) before vLLM init.
        ``post_llm``: one more pass after local vLLM spawns EngineCore so the parent
        process CUDA context is valid again before real inference.
        """
        if getattr(config, "sfm_device", "cuda") != "cuda":
            return
        if cls.asr is None:
            return

        if phase == "pre_llm":
            if getattr(config, "sfm_warmup", True) is False:
                return
            num_chunks = 2
            label = "Warming up ASR before LLM load"
        elif phase == "post_llm":
            if getattr(config, "sfm_warmup_after_llm", True) is False:
                return
            if getattr(config, "llm_base_url", None) is not None:
                return
            num_chunks = 1
            label = "Re-syncing ASR after vLLM init"
        else:
            raise ValueError(f"Unknown ASR warmup phase: {phase!r}")

        asr = cls.asr
        logger.info("%s on %s (%d chunk(s))...", label, asr.device, num_chunks)
        with torch.inference_mode():
            cls._run_asr_warmup_pass(asr, num_chunks=num_chunks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("%s complete", label)

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
                speculative=getattr(config, "sfm_speculative", False),
                speculative_max_words=getattr(config, "sfm_speculative_max_words", 0),
                pac_use_alignment_boundary=getattr(config, "pac_use_alignment_boundary", True),
                pac_emit_provisional_target=getattr(config, "pac_emit_provisional_target", True),
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
            cls._warmup_asr_model(config, phase="pre_llm")

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
            cls._warmup_asr_model(config, phase="post_llm")

        if getattr(config, "sfm_speculative", False):
            if not hasattr(cls, "word_aligner") or cls.word_aligner is None:
                cls.word_aligner = WordAligner(device="cpu", alignment_model="xlmr")
        else:
            cls.word_aligner = None

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.load_model(config)

        self.source_lang = getattr(config, "source_lang", "English")
        self.target_lang = getattr(config, "target_lang", "German")
        self._mt_emission_unit_locked = (
            getattr(config, "mt_word_level", None) is not None
            or getattr(config, "latency_unit", None) is not None
        )
        self._resolve_mt_emission_unit(config, self.target_lang)
        explicit_latency = getattr(config, "latency_unit", None)
        if explicit_latency is not None:
            self.latency_unit = explicit_latency
            if explicit_latency == "char":
                self._mt_word_level = False
                self._spaced_target = False
                self.target_sep = ""
            elif explicit_latency in {"word", "spm"}:
                self._mt_word_level = True
                self._spaced_target = True
                self.target_sep = " "
        else:
            self.latency_unit = "word" if self._mt_word_level else "char"

        self.min_start_seconds = getattr(config, "min_start_seconds", 1.0)
        self._temperature = getattr(config, "temperature", 0.7)
        self._top_p = getattr(config, "top_p", 0.9)
        self._top_k = getattr(config, "top_k", 20)
        self._max_tokens = getattr(config, "max_new_tokens", 256)
        self._repetition_penalty = getattr(config, "repetition_penalty", 1.05)
        # MT decoding strategy. "beam" fixes a deterministic beam search (recommended
        # for PAC experiments); "sampling" keeps the legacy temperature/top-p sampling.
        self._llm_decoding = str(getattr(config, "llm_decoding", "sampling")).lower()
        self._llm_beam_size = int(getattr(config, "llm_beam_size", 5))
        self._llm_beam_length_penalty = float(
            getattr(config, "llm_beam_length_penalty", 1.0)
        )
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
        # PAC variant: expose committed vs provisional ASR to the LLM as a tagged
        # prompt instead of a single flat source string.
        self._pac_tagged_prompt = bool(getattr(config, "pac_tagged_prompt", False))
        self._audio_noise_snr_db = getattr(config, "audio_noise_snr_db", None)
        self._audio_noise_rng = np.random.default_rng(
            int(getattr(config, "audio_noise_seed", 13))
        )

        # MT emission policy (LCP / LACP / SLCP), independent of ASR ``sfm_policy``.
        self._mt_policy = str(getattr(config, "mt_policy", "LCP")).upper()
        if self._mt_policy not in {"LCP", "LACP", "SLCP"}:
            raise ValueError(
                f"Unsupported mt_policy={self._mt_policy!r}; expected LCP, LACP, or SLCP"
            )
        self._mt_lacp_threshold = float(getattr(config, "mt_lacp_threshold", 2.0))
        self._mt_slcp_semantic_threshold = float(
            getattr(config, "mt_slcp_semantic_threshold", 0.65)
        )
        self._mt_slcp_max_gap = int(getattr(config, "mt_slcp_max_gap", 3))
        self._mt_slcp_use_spacy = bool(getattr(config, "mt_slcp_use_spacy", False))
        self._mt_slcp_nlp = None
        self._mt_slcp_linguistic_checks = None
        if self._mt_policy == "SLCP":
            if not _MT_SLCP_AVAILABLE:
                raise ImportError(
                    "mt_policy=SLCP requires segfreetk.states.sclp (_slcp_stable_flags); "
                    "install segfreetk and its SLCP dependencies"
                )
            self._ensure_mt_slcp_nlp()
        logger.info(
            "MT emission unit: %s (mt_word_level=%s, latency_unit=%s)",
            "word" if self._mt_word_level else "character",
            self._mt_word_level,
            self.latency_unit,
        )
        logger.info(
            "MT emission policy: %s (lacp_threshold=%s, slcp_semantic_threshold=%s, "
            "slcp_max_gap=%s, slcp_use_spacy=%s)",
            self._mt_policy,
            self._mt_lacp_threshold,
            self._mt_slcp_semantic_threshold,
            self._mt_slcp_max_gap,
            self._mt_slcp_use_spacy,
        )

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
        self._speech_chunk_size = float(speech_chunk_size) if speech_chunk_size is not None else None
        self._expected_input_chunk_samples = None
        if speech_chunk_size is not None:
            # Expected samples for a "full" SimulStream chunk. The final chunk
            # is often shorter; we use this to better set NeMo's last-chunk flag.
            self._expected_input_chunk_samples = int(round(float(speech_chunk_size) * self._input_sample_rate))

        self._saw_last_nonempty_chunk = False
        self._chunk_step = 0
        self._eval_progress = None
        if self._speech_chunk_size is not None:
            self._eval_progress = build_eval_progress_index(
                self._speech_chunk_size, config=config
            )
        logger.info(
            "Audio sample rates: simulstream=%s Hz, asr=%s Hz, resample=%s",
            self._input_sample_rate,
            self._asr_sample_rate,
            self._needs_resample,
        )

    def _resolve_mt_emission_unit(self, config: SimpleNamespace, target_lang: str) -> None:
        """Resolve MT emission granularity (word vs character) from config / target language."""
        explicit = getattr(config, "mt_word_level", None)
        if explicit is not None:
            mt_word_level = bool(explicit)
        else:
            latency_unit = getattr(config, "latency_unit", None)
            if latency_unit == "char":
                mt_word_level = False
            elif latency_unit in {"word", "spm"}:
                mt_word_level = True
            else:
                mt_word_level = not _is_char_based_target_lang(target_lang)

        self._mt_word_level = mt_word_level
        self._spaced_target = mt_word_level
        self.target_sep = " " if mt_word_level else ""

    def _apply_mt_emission_unit_for_language(self, target_lang: str) -> None:
        if self._mt_emission_unit_locked:
            return
        self._mt_word_level = not _is_char_based_target_lang(target_lang)
        self._spaced_target = self._mt_word_level
        self.target_sep = " " if self._mt_word_level else " "
        self.latency_unit = "word" if self._mt_word_level else "char"

    def _join_translation_parts(self, prev_prefix: str, hypothesis: str) -> str:
        prev = prev_prefix.strip()
        hyp = hypothesis.strip()
        if not prev:
            return hyp
        if not hyp:
            return prev
        if self._mt_word_level:
            return f"{prev} {hyp}"
        return f"{prev}{hyp}"

    def _format_mt_tokens_for_log(self, text: str) -> str:
        tokens = self._text_to_tokens(text)
        if not tokens:
            return ""
        return self.tokens_to_string(tokens)

    def _mt_tokenize(self, text: str) -> List[str]:
        return self._text_to_tokens(text)

    def _maybe_add_audio_noise(self, waveform: np.ndarray) -> np.ndarray:
        if self._audio_noise_snr_db is None or waveform is None or len(waveform) == 0:
            return waveform
        waveform64 = waveform.astype(np.float64, copy=False)
        signal_power = float(np.mean(np.square(waveform64)))
        if signal_power <= 0.0:
            return waveform
        snr_linear = 10.0 ** (float(self._audio_noise_snr_db) / 10.0)
        noise_std = np.sqrt(signal_power / snr_linear)
        noise = self._audio_noise_rng.normal(0.0, noise_std, size=waveform.shape)
        return (waveform + noise).astype(np.float32)

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
    ) -> tuple[str, str]:
        speculative = getattr(self.asr_cfg, "speculative", False)

        if (waveform is None or len(waveform) == 0) and not is_last_chunk:
            logger.warning(f"[ASR] Received empty waveform. Returning empty string.")
            return "", ""

        if waveform is not None and len(waveform) > 0:
            waveform = self._maybe_add_audio_noise(waveform)
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
            flush_result = state.asr_hyp_buffer.flush(last_instant=0, speculative=speculative)
        elif self.asr_cfg.policy == "LACP":
            flush_result = state.asr_hyp_buffer.flush(forced=True, speculative=speculative)
        elif self.asr_cfg.policy == "LCP" and state.nchunks_no_output >= max_empty_chunks:
            flush_result = state.asr_hyp_buffer.flush(forced=True, speculative=speculative)
        else:
            flush_result = state.asr_hyp_buffer.flush(speculative=speculative)

        if speculative:
            logger.debug(f"[ASR] Speculative flush result with policy {self.asr_cfg.policy}: {flush_result}")
            out, spec_out = flush_result
        else:
            out = flush_result
            spec_out = []

        if is_last_chunk:
            out.extend(state.asr_hyp_buffer.complete())
            spec_out = []

        if max_empty_chunks:
            if not out:
                state.nchunks_no_output += 1
            else:
                state.nchunks_no_output = 0

        if not out and not spec_out:
            logger.info(f"[ASR] {self.asr_cfg.policy} policy generated no transcription. Emitting empty string")
            return "", ""

        res = self._tokens_to_text([t for _, _, t in out]) if out else ""
        spec = self._tokens_to_text([t for _, _, t in spec_out]) if spec_out else ""
        max_spec_words = int(getattr(self.asr_cfg, "speculative_max_words", 0) or 0)
        if spec and max_spec_words > 0:
            spec_words = spec.split()
            if len(spec_words) > max_spec_words:
                spec = " ".join(spec_words[:max_spec_words])
                logger.info(
                    "[ASR] Capped provisional ASR context to %d word(s): %r",
                    max_spec_words,
                    spec,
                )
        logger.info(f"[ASR] Emitting committed '{res}', speculative '{spec}'")
        return res, spec

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

    def _build_llm_prompt(
        self,
        asr_segment: str,
        prev_translation: str,
        speculative_segment: Optional[str] = None,
    ) -> str:
        use_tagged = (
            self._pac_tagged_prompt
            and speculative_segment is not None
            and speculative_segment.strip() != ""
        )
        if use_tagged:
            system_content = (
                f"You are a professional simultaneous speech translator. "
                f"Translate from {self.source_lang} into {self.target_lang}. "
                "The source is given in two parts: a COMMITTED section that is final, "
                "and a TENTATIVE section transcribed from partial audio that may still "
                "change. Translate the committed section faithfully and use the tentative "
                "section as additional context to produce a coherent, well-ordered "
                "translation of the whole source. "
                "Preserve named entities exactly as in the source text. "
                "Output only the translation with no explanation, preamble, or reasoning."
            )
            user_content = (
                f"COMMITTED:\n{asr_segment.strip()}\n\n"
                f"TENTATIVE:\n{speculative_segment.strip()}"
            )
        else:
            system_content = (
                f"You are a professional simultaneous speech translator. "
                f"Translate from {self.source_lang} into {self.target_lang}. "
                "Preserve named entities exactly as in the source text. "
                "Output only the translation with no explanation, preamble, or reasoning."
            )
            user_content = asr_segment
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
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

    def _fit_llm_prompt(
        self,
        asr_segment: str,
        prev_translation: str,
        speculative_segment: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """
        Build an LLM prompt that fits within the configured context window.

        Long ACL segments can exceed ``llm_max_model_len`` once ASR and the committed
        translation prefix grow; drop older ASR first, then older translation prefix.

        When ``speculative_segment`` is provided (tagged PAC), only the committed
        ``asr_segment`` is truncated; the short provisional suffix is kept intact.
        """
        reserve_tokens = self._max_tokens + 32
        max_input_tokens = max(64, self._llm_max_model_len - reserve_tokens)

        asr_words = asr_segment.split()
        prev = self._normalize_translation_text(prev_translation)
        prompt = self._build_llm_prompt(asr_segment, prev, speculative_segment)
        prompt_tokens = self._count_prompt_tokens(prompt)

        if prompt_tokens > max_input_tokens and len(asr_words) > 1:
            lo, hi = 0, len(asr_words)
            best_asr = asr_segment
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = " ".join(asr_words[mid:])
                candidate_prompt = self._build_llm_prompt(candidate, prev, speculative_segment)
                if self._count_prompt_tokens(candidate_prompt) <= max_input_tokens:
                    best_asr = candidate
                    hi = mid
                else:
                    lo = mid + 1
            asr_segment = best_asr
            prompt = self._build_llm_prompt(asr_segment, prev, speculative_segment)
            prompt_tokens = self._count_prompt_tokens(prompt)
            if lo > 0:
                logger.warning(
                    "Truncated ASR prompt from %d to %d words to fit %d input tokens",
                    len(asr_words),
                    len(asr_segment.split()),
                    max_input_tokens,
                )

        if prompt_tokens > max_input_tokens and prev:
            template = self._build_llm_prompt(asr_segment, "", speculative_segment)
            template_tokens = self._count_prompt_tokens(template)
            prev_budget = max(0, max_input_tokens - template_tokens)
            prev = self._normalize_translation_text(
                self._truncate_text_from_left(prev, prev_budget)
            )
            prompt = self._build_llm_prompt(asr_segment, prev, speculative_segment)
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
        if not self._mt_word_level:
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
        state.mt_slcp_prev_hyp = self._text_to_tokens(state.prev_translation)
        logger.warning(
            "[BAD STATE] Rolled back last %d translation unit(s); committed prefix is now %r",
            n_units,
            state.prev_translation,
        )

    def _reset_translation_state(self, state: CascadeState) -> None:
        state.prev_translation = ""
        state.translation_hypotheses = [""]
        state.mt_slcp_prev_hyp = []
        state.consecutive_empty_mt = 0

    def _ensure_mt_slcp_nlp(self) -> None:
        """Lazily load spacy for SLCP Pass 3 linguistic checks (segfreetk-compatible)."""
        if self._mt_slcp_nlp is not None or not self._mt_slcp_use_spacy:
            return
        try:
            import spacy

            self._mt_slcp_nlp = spacy.load("en_core_web_sm")
            self._mt_slcp_linguistic_checks = _LinguisticChecks(
                lemma=True, pos_dep=True, ner=True, fine_tag=True
            )
            logger.info("[MT/SLCP] spacy model loaded (en_core_web_sm)")
        except ImportError:
            logger.warning(
                "[MT/SLCP] spacy unavailable; falling back to morphological similarity"
            )
        except OSError as exc:
            logger.warning(
                "[MT/SLCP] spacy model unavailable (%s); falling back to morphological similarity",
                exc,
            )

    def _mt_slcp_stable_prefix(
        self, state: CascadeState, curr_hypothesis: str, prev_prefix: str
    ) -> str:
        """Stable-LCP emission for MT (segfreetk TranslationState.SLCP)."""
        assert _slcp_stable_flags is not None

        prev_hyp_tokens = state.mt_slcp_prev_hyp
        curr_tokens = self._text_to_tokens(curr_hypothesis)
        cursor = len(self._text_to_tokens(prev_prefix))

        if not prev_hyp_tokens:
            return prev_prefix

        stable_flags = _slcp_stable_flags(
            prev_tokens=prev_hyp_tokens,
            curr_tokens=curr_tokens,
            semantic_threshold=self._mt_slcp_semantic_threshold,
            nlp=self._mt_slcp_nlp,
            linguistic_checks=self._mt_slcp_linguistic_checks if self._mt_slcp_nlp else None,
            max_gap=self._mt_slcp_max_gap,
        )

        new_tokens: List[str] = []
        for i in range(cursor, len(curr_tokens)):
            if i < len(stable_flags) and stable_flags[i]:
                new_tokens.append(curr_tokens[i])
            else:
                break

        if not new_tokens:
            return prev_prefix

        committed_tokens = self._text_to_tokens(prev_prefix) + new_tokens
        return self._normalize_translation_text(self.tokens_to_string(committed_tokens))

    def _compute_mt_stable_prefix(
        self,
        state: CascadeState,
        prev_hypothesis: str,
        curr_hypothesis: str,
        prev_prefix: str,
    ) -> str:
        if self._mt_policy == "LCP":
            return self._normalize_translation_text(
                longest_common_prefix(
                    prev_hypothesis, curr_hypothesis, word_level=self._mt_word_level
                )
            )
        if self._mt_policy == "LACP":
            prev_tokens = self._text_to_tokens(prev_hypothesis)
            curr_tokens = self._text_to_tokens(curr_hypothesis)
            stable_tokens = word_lacp_prefix_tokens(
                prev_tokens,
                curr_tokens,
                self._mt_lacp_threshold,
                word_level=self._mt_word_level,
            )
            return self._normalize_translation_text(self.tokens_to_string(stable_tokens))
        if self._mt_policy == "SLCP":
            return self._mt_slcp_stable_prefix(state, curr_hypothesis, prev_prefix)
        raise ValueError(f"Unsupported mt_policy: {self._mt_policy}")

    def _llm_beam_search(self, prompt: str, max_tokens: int) -> str:
        """Deterministic beam search decoding for both local and remote vLLM backends."""
        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={
                    "use_beam_search": True,
                    "best_of": self._llm_beam_size,
                    "length_penalty": self._llm_beam_length_penalty,
                    "chat_template_kwargs": {"enable_thinking": self._llm_enable_thinking},
                },
            )
            logger.info(f"[LLM] Beam response: '{response.choices[0].text}'")
            return self._sanitize_llm_output(response.choices[0].text)

        from vllm.sampling_params import BeamSearchParams

        beam_params = BeamSearchParams(
            beam_width=self._llm_beam_size,
            max_tokens=max_tokens,
            length_penalty=self._llm_beam_length_penalty,
        )
        outputs = self.llm.beam_search([prompt], beam_params)
        sequences = outputs[0].sequences
        best = sequences[0].text if sequences else ""
        llm_out = self._sanitize_llm_output(best)
        logger.info(f"[LLM] Beam response: '{llm_out}'")
        return llm_out

    def _llm_generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        #logger.debug(f"[LLM] Prompt: '{prompt}'")
        prompt_tokens = self._count_prompt_tokens(prompt)
        max_tokens = min(self._max_tokens, max(1, self._llm_max_model_len - prompt_tokens - 1))

        if self._llm_decoding == "beam":
            return self._llm_beam_search(prompt, max_tokens)

        temp = self._temperature if temperature is None else temperature

        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self._top_p,
                extra_body={
                    "repetition_penalty": self._repetition_penalty,
                    "chat_template_kwargs": {"enable_thinking": self._llm_enable_thinking},
                },
            )
            logger.info(f"[LLM] Response: '{response.choices[0].text}'")
            return self._sanitize_llm_output(response.choices[0].text)
        sampling_params = SamplingParams(
            temperature=temp,
            top_p=self._top_p,
            top_k=self._top_k,
            max_tokens=max_tokens,
            repetition_penalty=self._repetition_penalty,
        )
        llm_outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        logger.debug(f"[LLM] Pre-sanitiized response: {self.tokenizer.convert_ids_to_tokens(llm_outputs[0].outputs[0].token_ids)}")
        llm_out = self._sanitize_llm_output(llm_outputs[0].outputs[0].text)
        logger.info(f"[LLM] Response: '{llm_out}'")
        return llm_out

    def _llm_generate_with_fallback(
        self,
        state: CascadeState,
        asr_text: str,
        prev_prefix: str,
        speculative_segment: Optional[str] = None,
    ) -> str:
        """
        MT generation with Whisper-style fallbacks (segfreetk vllm.py):
        - empty output: after N consecutive empties, roll back committed translation and retry
        - repetition loop: retry with rising temperature until compression ratio drops
        """
        prompt, asr_text, prev_prefix = self._fit_llm_prompt(
            asr_text, prev_prefix, speculative_segment
        )
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
                prompt, _, prev_prefix = self._fit_llm_prompt(
                    asr_text, state.prev_translation, speculative_segment
                )
                hypothesis = self._llm_generate(prompt)
                if not hypothesis.strip():
                    logger.warning(
                        "[BAD STATE] No MT output after empty-generation rollback; "
                        "keeping previous translation prefix"
                    )
            return hypothesis

        state.consecutive_empty_mt = 0

        # Beam search is deterministic; rising-temperature retries would re-run the
        # identical decode, so skip the repetition fallback loop under beam decoding.
        if not self._temp_fall or self._llm_decoding == "beam":
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

    def _align_translation_to_source(
        self, state: CascadeState, source_text: str, translation_text: str
    ) -> Optional[AlignmentResult]:
        if self.word_aligner is None:
            return None
        alignment = self.word_aligner.align(
            WordAligner.tokenize(source_text),
            self._mt_tokenize(translation_text),
        )
        state.last_alignment = alignment
        logger.debug(
            "[Align] method=%s pairs=%s",
            alignment.alignment_method,
            alignment.alignments,
        )
        return alignment

    def _first_speculative_tgt_word_idx(
        self, committed_text: str, alignment: AlignmentResult
    ) -> Optional[int]:
        """Return the first target-word index aligned to speculative source tokens."""
        committed_count = len(WordAligner.tokenize(committed_text)) if committed_text else 0
        speculative_tgt_idxs = [
            tgt_idx for src_idx, tgt_idx in alignment.alignments if src_idx >= committed_count
        ]
        if not speculative_tgt_idxs:
            return None
        return min(speculative_tgt_idxs)

    def _split_translation_at_word_idx(self, text: str, word_idx: Optional[int]) -> tuple[str, str]:
        tokens = self._mt_tokenize(text)
        if word_idx is None or word_idx >= len(tokens):
            return text, ""
        stable = self.tokens_to_string(tokens[:word_idx])
        speculative = self.tokens_to_string(tokens[word_idx:])
        return stable, speculative

    def _update_speculative_translation_boundary(
        self,
        state: CascadeState,
        committed_text: str,
        translation_text: str,
        alignment: Optional[AlignmentResult],
    ) -> None:
        if alignment is None or not state.speculative_text.strip():
            state.first_speculative_word_idx = None
            state.current_speculative_tokens = []
            return

        state.first_speculative_word_idx = self._first_speculative_tgt_word_idx(
            committed_text, alignment
        )
        stable_part, speculative_part = self._split_translation_at_word_idx(
            translation_text, state.first_speculative_word_idx
        )
        state.current_speculative_tokens = WordAligner.tokenize(speculative_part)
        logger.info(
            "[MT] Speculative boundary at word %s: speculative=%r",
            state.first_speculative_word_idx,
            speculative_part,
        )

    @staticmethod
    def _strip_reemitted_speculative_stable(
        stable_increment_tokens: List[str], last_spec: List[str]
    ) -> List[str]:
        """Drop stable tokens that were already emitted as speculative."""
        if not stable_increment_tokens or not last_spec:
            return stable_increment_tokens

        validated = word_lcp_len(last_spec, stable_increment_tokens)
        stable_tokens = stable_increment_tokens[validated:]
        if not stable_tokens:
            return stable_tokens

        if stable_tokens == last_spec:
            return []
        if len(stable_tokens) <= len(last_spec) and stable_tokens == last_spec[-len(stable_tokens) :]:
            return []
        if len(stable_tokens) >= len(last_spec) and stable_tokens[-len(last_spec) :] == last_spec:
            return stable_tokens[: -len(last_spec)]

        return stable_tokens

    @staticmethod
    def _strip_overlap_with_last_emission(
        stable_tokens: List[str], last_emitted: List[str]
    ) -> List[str]:
        """Drop a stable prefix that repeats the tail of the previous chunk emission."""
        if not stable_tokens or not last_emitted:
            return stable_tokens

        max_overlap = min(len(stable_tokens), len(last_emitted))
        for size in range(max_overlap, 0, -1):
            if stable_tokens[:size] == last_emitted[-size:]:
                return stable_tokens[size:]
        return stable_tokens

    def _filter_stable_increment_tokens(
        self,
        state: CascadeState,
        stable_increment_tokens: List[str],
    ) -> List[str]:
        filtered = self._strip_reemitted_speculative_stable(
            stable_increment_tokens, state.last_emitted_speculative_tokens
        )
        filtered = self._strip_overlap_with_last_emission(filtered, state.last_emitted_tokens)
        if filtered != stable_increment_tokens:
            logger.info(
                "[MT] Skipping re-emitted stable tokens: %r -> %r",
                stable_increment_tokens,
                filtered,
            )
        return filtered

    def _speculative_stream_tail(self, state: CascadeState) -> List[str]:
        n = state.stream_speculative_suffix_len
        if n <= 0:
            return []
        return state.cumulative_emitted_tokens[-n:]

    def _update_cumulative_emission(
        self,
        state: CascadeState,
        deleted_tokens: List[str],
        new_tokens: List[str],
        new_spec_len: int,
    ) -> None:
        if deleted_tokens:
            n = len(deleted_tokens)
            stream_tail = state.cumulative_emitted_tokens[-n:]
            if stream_tail != deleted_tokens:
                logger.warning(
                    "[MT] Stream tail mismatch during update: deleted=%r tail=%r",
                    deleted_tokens,
                    stream_tail,
                )
            state.cumulative_emitted_tokens = state.cumulative_emitted_tokens[:-n]
        state.cumulative_emitted_tokens.extend(new_tokens)
        state.stream_speculative_suffix_len = new_spec_len

    def _compute_speculative_emission_delta(
        self,
        state: CascadeState,
        stable_increment_tokens: List[str],
    ) -> tuple[List[str], List[str], int]:
        """
        Compare the current speculative translation with the stream tail.

        Stable increments can promote a prefix of the speculative stream suffix to
        committed text. Promoted tokens stay on the cumulative stream; only the
        remaining unvalidated speculative suffix may be deleted.
        """
        spec_tokens = state.current_speculative_tokens
        stable_to_emit = self._filter_stable_increment_tokens(state, stable_increment_tokens)
        stream_spec_tail = self._speculative_stream_tail(state)

        promoted_len = word_lcp_len(stream_spec_tail, stable_increment_tokens)
        if promoted_len:
            logger.info(
                "[MT] Promoted %d speculative token(s) to stable: %r",
                promoted_len,
                stream_spec_tail[:promoted_len],
            )
        unvalidated_stream_spec = stream_spec_tail[promoted_len:]

        if unvalidated_stream_spec == spec_tokens:
            if stable_to_emit:
                deleted_tokens = list(unvalidated_stream_spec)
                new_spec_tokens = list(spec_tokens)
            else:
                deleted_tokens = []
                new_spec_tokens = []
        elif (
            not stable_to_emit
            and word_lcp_len(unvalidated_stream_spec, spec_tokens) == len(unvalidated_stream_spec)
            and len(spec_tokens) >= len(unvalidated_stream_spec)
        ):
            deleted_tokens = []
            new_spec_tokens = spec_tokens[len(unvalidated_stream_spec) :]
        else:
            deleted_tokens = list(unvalidated_stream_spec)
            new_spec_tokens = list(spec_tokens)

        if deleted_tokens:
            new_spec_len = len(new_spec_tokens)
        elif new_spec_tokens:
            new_spec_len = len(unvalidated_stream_spec) + len(new_spec_tokens)
        else:
            new_spec_len = len(unvalidated_stream_spec)

        state.last_emitted_speculative_tokens = list(spec_tokens)
        new_tokens = stable_to_emit + new_spec_tokens
        return new_tokens, deleted_tokens, new_spec_len

    def _translate_from_asr(self, state: CascadeState, force_final: bool) -> str:
        asr_text = state.asr_committed_text.strip()
        speculative_text = state.speculative_text.strip()
        if not asr_text and not speculative_text:
            return ""

        speculative = bool(speculative_text)
        asr_context = f"{asr_text} {speculative_text}".strip() if speculative else asr_text

        prev_prefix = self._normalize_translation_text(state.prev_translation)
        if speculative and self._pac_tagged_prompt:
            hypothesis = self._llm_generate_with_fallback(
                state, asr_text, prev_prefix, speculative_segment=speculative_text
            )
        else:
            hypothesis = self._llm_generate_with_fallback(state, asr_context, prev_prefix)
        prev_prefix = self._normalize_translation_text(state.prev_translation)
        full_hypothesis = self._normalize_translation_text(
            self._join_translation_parts(prev_prefix, hypothesis)
        )

        use_boundary = bool(getattr(self.asr_cfg, "pac_use_alignment_boundary", True))
        emit_provisional_target = bool(
            getattr(self.asr_cfg, "pac_emit_provisional_target", True)
        )

        if speculative and use_boundary:
            alignment = self._align_translation_to_source(state, asr_context, full_hypothesis)
            self._update_speculative_translation_boundary(
                state, asr_text, full_hypothesis, alignment
            )
            if not emit_provisional_target:
                stable_part, _ = self._split_translation_at_word_idx(
                    full_hypothesis, state.first_speculative_word_idx
                )
                stable_part = self._normalize_translation_text(stable_part)
                if prev_prefix and not stable_part.startswith(prev_prefix):
                    stable_part = prev_prefix
                full_hypothesis = stable_part
                state.current_speculative_tokens = []
        elif speculative:
            state.last_alignment = None
            state.first_speculative_word_idx = None
            state.current_speculative_tokens = []
        else:
            state.last_alignment = None
            state.first_speculative_word_idx = None
            state.current_speculative_tokens = []

        logger.debug(f"[LLM] Prev prefix: {prev_prefix}")
        logger.debug(f"[LLM] Hypothesis: {hypothesis}")
        #logger.debug(f"[LLM] Full hypothesis: {full_hypothesis}")

        if force_final:
            logger.debug(f"[LLM] Force final")
            increment = full_hypothesis[len(prev_prefix) :]
            state.prev_translation = full_hypothesis
            state.current_speculative_tokens = []
            return self._trim_translation_increment(increment)

        state.translation_hypotheses.append(full_hypothesis)
        stable = self._compute_mt_stable_prefix(
            state,
            state.translation_hypotheses[-2],
            state.translation_hypotheses[-1],
            prev_prefix,
        )
        if self._mt_policy == "SLCP":
            state.mt_slcp_prev_hyp = self._text_to_tokens(full_hypothesis)
        logger.info(
            "[MT/%s] Stable: %r",
            self._mt_policy,
            self._format_mt_tokens_for_log(stable[len(prev_prefix) :]),
        )
        increment = stable[len(prev_prefix) :]
        state.prev_translation = stable
        return self._trim_translation_increment(increment)

    def _text_to_tokens(self, text: str) -> List[str]:
        if text == "":
            return []
        text = self._normalize_translation_text(text)
        if self._mt_word_level:
            return [tok for tok in text.strip().split() if tok]
        return list(text.strip())

    def _build_incremental_output(self, stable_increment: str) -> IncrementalOutput:
        stable_increment = self._trim_translation_increment(
            self._normalize_translation_text(stable_increment or "")
        )
        stable_tokens = self._text_to_tokens(stable_increment)

        if getattr(self.asr_cfg, "speculative", False) and getattr(
            self.asr_cfg, "pac_emit_provisional_target", True
        ):
            new_tokens, deleted_tokens, new_spec_len = self._compute_speculative_emission_delta(
                self._state, stable_tokens
            )
        else:
            new_tokens = stable_tokens
            deleted_tokens = []
            new_spec_len = 0

        if not new_tokens and not deleted_tokens:
            return IncrementalOutput([], "", [], "")

        new_string = self.tokens_to_string(new_tokens)
        if (
            self._mt_word_level
            and self._state.emission_started
            and new_string
            and not new_string.startswith(" ")
        ):
            new_string = " " + new_string

        deleted_string = self.tokens_to_string(deleted_tokens) if deleted_tokens else ""
        if new_tokens:
            self._state.emission_started = True

        if deleted_tokens:
            logger.info(
                "[MT] Retracting speculative tokens %r (stream tail %r); emitting %r",
                deleted_string,
                self._speculative_stream_tail(self._state),
                new_string,
            )
            logger.info(f"[MT] Cummulative emitted tokens is now: {self._state.cumulative_emitted_tokens}")

        self._update_cumulative_emission(
            self._state, deleted_tokens, new_tokens, new_spec_len
        )
        if new_tokens or deleted_tokens:
            self._state.last_emitted_tokens = list(new_tokens)

        return IncrementalOutput(
            new_tokens=new_tokens,
            new_string=new_string,
            deleted_tokens=deleted_tokens,
            deleted_string=deleted_string,
        )

    @torch.inference_mode()
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if waveform is None or len(waveform) == 0:
            return IncrementalOutput([], "", [], "")

        self._chunk_step += 1
        logger.info(
            format_step_banner(
                self._eval_progress, self._state.speech_id, self._chunk_step
            )
        )

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

        asr_increment, asr_speculative = self._asr_step(
            self._state, waveform, is_last_chunk=is_last_chunk
        )
        self._state.speculative_text = asr_speculative.strip()
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
        asr_increment, asr_speculative = self._asr_step(
            self._state,
            np.zeros(0, dtype=np.float32),
            is_last_chunk=not self._saw_last_nonempty_chunk,
        )
        self._state.speculative_text = asr_speculative.strip()
        if asr_increment:
            if self._state.asr_committed_text:
                self._state.asr_committed_text = f"{self._state.asr_committed_text} {asr_increment}".strip()
            else:
                self._state.asr_committed_text = asr_increment.strip()

        translation = self._translate_from_asr(self._state, force_final=True)
        current_speech_id = self._state.speech_id + 1
        self._state = self._fresh_state(speech_id=current_speech_id)
        self._chunk_step = 0
        self._saw_last_nonempty_chunk = False
        return self._build_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        self.source_lang = language

    def set_target_language(self, language: str) -> None:
        self.target_lang = language
        self._apply_mt_emission_unit_for_language(language)

    def tokens_to_string(self, tokens: List[str]) -> str:
        if self._mt_word_level:
            return " ".join(tokens)
        return "".join(tokens)

    def clear(self) -> None:
        self._state = self._fresh_state(speech_id=self._state.speech_id)
        self._chunk_step = 0
        self._saw_last_nonempty_chunk = False
