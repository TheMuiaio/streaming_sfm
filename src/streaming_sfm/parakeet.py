from simuleval.agents import SpeechToTextAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.utils import entrypoint

from streaming_sfm.streaming_model import StreamingParakeet, StreamingBatchedAudioBufferWithOffset
from streaming_sfm.hyp_utils import (
    SLCPHypothesisBuffer,
    ABSHypothesisBuffer,
    LCPHypothesisBuffer,
    LACPHypothesisBuffer,
    WaitKHypothesisBuffer,
    HoldNHypothesisBuffer,
)

import torch

from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass
from argparse import Namespace, ArgumentParser

from typing import Optional

import logging
logger = logging.getLogger(__name__)

from streaming_sfm import LOG_LEVEL
logger.setLevel(LOG_LEVEL)


@dataclass
class ParakeetStreamingStates(AgentStates):
    buffer: StreamingBatchedAudioBufferWithOffset
    hyp_buffer: ABSHypothesisBuffer
    current_offset: int
    encoder_frame2audio_samples: int
    left_sample: int
    right_sample: int
    nchunks_no_output: int
    incomplete_buffer: list
    dtype: torch.dtype
    device: torch.device
    debug: False

    def reset(self, cfg, asr_model):
        super().reset()

        self.buffer = self.reset_audio_buffer(asr_model)
        self.hyp_buffer = self.reset_hyp_buffer(cfg, asr_model)
        self.current_offset = 0
        self.encoder_frame2audio_samples = asr_model.encoder_frame2audio_samples
        self.left_sample = 0
        self.right_sample = asr_model.context_samples.chunk + asr_model.context_samples.right
        self.nchunks_no_ouptut = 0
        self.incomplete_buffer = []
        self.dtype = asr_model.dtype
        self.device = asr_model.device
        self.speech_id = 0

    def reset_hyp_buffer(self, cfg, model):
        word_level = getattr(cfg, 'word_level', False)
        if cfg.policy == 'LCP':
            hyp_buffer = LCPHypothesisBuffer(word_level=word_level, debug=self.debug)
        elif cfg.policy == 'LACP':
            hyp_buffer = LACPHypothesisBuffer(cfg.lacp_threshold, word_level=word_level, debug=self.debug)
        elif cfg.policy == 'SLCP':
            hyp_buffer = _build_slcp_buffer(cfg, word_level=word_level, debug=self.debug)
        elif cfg.policy == 'WaitK':
            hyp_buffer = WaitKHypothesisBuffer(
                cfg.K,
                features_per_second=model.features_per_sec,
                subsampling_factor=model.subsampling_factor,
                word_level=word_level,
                debug=self.debug,
            )
        else:
            hyp_buffer = HoldNHypothesisBuffer(cfg.N, word_level=word_level, debug=self.debug)
        return hyp_buffer

    def reset_audio_buffer(self, asr_model):
        buffer_len = self.buffer.get_buffer_len()[1]
        self.current_offset += (buffer_len // self.encoder_frame2audio_samples)

        return StreamingBatchedAudioBufferWithOffset(
            batch_size=1,
            context_samples=asr_model.context_samples,
            dtype=asr_model.dtype,
            device=asr_model.device,
        )

    def update_source(self, segment):
        if segment.finished:
            self.source_finished = segment.finished
        stride = self.buffer.add_audio_batch_get_stride(
            torch.tensor([segment.content], device=self.device),
            audio_lengths=torch.tensor([len(segment.content)], device=self.device),
            is_last_chunk=segment.finished,
            is_last_chunk_batch=torch.tensor([segment.finished], device=self.device),
        )
        self.current_offset += (stride // self.encoder_frame2audio_samples)
        super().update_source(segment)


import logging
logger = logging.getLogger(__name__)
try:
    from segfreetk import LOG_LEVEL
    logger.setLevel(LOG_LEVEL)
except Exception:
    logger.info("Segfreetk not installed.")

from types import SimpleNamespace



def _build_slcp_buffer(cfg, word_level: bool = False, debug: bool = False):
    """
    Construct an SLCPHypothesisBuffer from an OmegaConf/Namespace cfg.

    Optionally loads a spacy model when --sfm_slcp_use_spacy is set.
    The spacy model is loaded once here; passing it around avoids reloading
    it on every sentence reset.
    """
    nlp = None
    linguistic_checks = None

    if getattr(cfg, 'slcp_use_spacy', False):
        try:
            import spacy
            from streaming_sfm.slcp_helpers import _LinguisticChecks
            nlp = spacy.load("en_core_web_sm")
            linguistic_checks = _LinguisticChecks()
            logger.info("[SLCP] spacy model loaded (en_core_web_sm)")
        except Exception as e:
            logger.warning(f"[SLCP] spacy unavailable, falling back to morphological similarity: {e}")

    return SLCPHypothesisBuffer(
        semantic_threshold=getattr(cfg, 'slcp_semantic_threshold', None),
        nlp=nlp,
        linguistic_checks=linguistic_checks,
        max_gap=getattr(cfg, 'slcp_max_gap', None),
        word_level=word_level,
        debug=debug,
    )


@entrypoint
class ParakeetAgent(SpeechToTextAgent):
    def __init__(self, args: SimpleNamespace):
        for key, value in vars(args).items():
            setattr(self, key, value)

        self.pdfs =  getattr(args, "pdfs", False)
        logger.debug(args)
        logger.debug(args.__dict__.keys())

        boosting_alpha = getattr(args, "sfm_boosting_tree_alpha", 1.0)
        beam_size = getattr(args, "sfm_decode", 1)
        logger.debug(f"Beam size will be {beam_size}")
        boosting_cfg = {
            "context_score": getattr(args, "sfm_context_score", 1.0),
            "depth_scaling": getattr(args, "sfm_depth_scaling", 2.0),
            #"key_phrases_file": None, #Unset so we dont overrideit it afterwards
        }
        decoding_cfg = {
            "strategy": "greedy_batch" if beam_size == 1 else "malsd_batch",
            "greedy": {"boosting_tree": boosting_cfg, "boosting_tree_alpha": boosting_alpha},
            "beam": {"boosting_tree": boosting_cfg, "boosting_tree_alpha": boosting_alpha, "beam_size": beam_size},
        }
        cfg_args = Namespace(
            model_path=getattr(args, "sfm_model_path", None),
            pretrained_name=getattr(args, "sfm_pretrained_name", "nvidia/parakeet-tdt-0.6b-v3"),
            manifest_path=getattr(args, "sfm_manifest_path", "vp.jsonl"),

            chunk_secs=getattr(args, "sfm_chunk_secs", 1),
            left_context_secs=getattr(args, "sfm_left_context_secs", 20),
            right_context_secs=getattr(args, "sfm_right_context_secs", 0),

            # ----------------------------------------------------------------
            # NEW: max_empty_chunks
            # ----------------------------------------------------------------
            max_empty_chunks=getattr(args, "sfm_max_empty_chunks", 0),

            policy=getattr(args, "sfm_policy", "LACP"),
            lacp_threshold=getattr(args, "sfm_lacp_threshold", 2),
            K=getattr(args, "sfm_K", 2),
            N=getattr(args, "sfm_N", 5),

            # ----------------------------------------------------------------
            # NEW: word_level flag
            # ----------------------------------------------------------------
            word_level=getattr(args, "sfm_word_level", False),

            # SLCP-specific
            slcp_semantic_threshold=getattr(args, "sfm_slcp_semantic_threshold", 0.65),
            slcp_max_gap=getattr(args, "sfm_slcp_max_gap", 3),
            slcp_use_spacy=getattr(args, "sfm_slcp_use_spacy", False),

            device=getattr(args, "sfm_device", "cuda"),
            compute_dtype=getattr(args, "sfm_compute_dtype", "bfloat16"),
            emit_incomplete=getattr(args, "sfm_emit_incomplete", False),
            rnnt_decoding=decoding_cfg,
        )
        self.cfg = OmegaConf.create(vars(cfg_args))
        with open_dict(self.cfg):
            self.cfg.cuda = 0 if cfg_args.device == "cuda" else -1
            self.cfg.allow_mps = True if cfg_args.device == "mps" else False

        model_id = cfg_args.pretrained_name
        self.cfg.model_path = None
        if not model_id:
            raise ValueError("Neither of --model_path or --pretrained_name were provided")
        print(f"--- Initializing Streaming Parakeet ---")

        #import tempfile
        #if getattr(args, "pdfs", False):
        #    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".txt") as word_boost_tmp:
        #        word_boost_list = "Markko Turchi\nSara Pape"
        #        word_boost_tmp.write(word_boost_list)
        #        word_boost_tmp.flush()
        #        self.cfg.rnnt_decoding.greedy.boosting_tree.key_phrases_file = word_boost_tmp.name
        #        self.cfg.rnnt_decoding.beam.boosting_tree.key_phrases_file = word_boost_tmp.name
        #        self.model = StreamingParakeet(self.cfg)
        #else:
        #TODO Double load with pdf
        self.model = StreamingParakeet(self.cfg, mbr=getattr(args, "sfm_mbr", False))
        super().__init__(args)

    def set_boosting_tree(self, pdf_entities):
        import tempfile
        if self.pdfs:
            with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".txt") as word_boost_tmp:
                word_boost_list ="\n".join(pdf_entities)
                word_boost_tmp.write(word_boost_list)
                word_boost_tmp.flush()

                logger.debug(self.model.asr_model.cfg.decoding)
                with open_dict(self.model.asr_model.cfg.decoding):
                    self.model.asr_model.cfg.decoding.greedy.boosting_tree.key_phrases_file = word_boost_tmp.name
                    self.model.asr_model.cfg.decoding.beam.boosting_tree.key_phrases_file = word_boost_tmp.name
                    self.model.asr_model.cfg.decoding.greedy.preserve_alignments = True
                    self.model.asr_model.cfg.decoding.beam.preserve_alignments = True
                    self.model.asr_model.cfg.decoding.tdt_include_token_duration = True
                logger.debug(self.model.asr_model.cfg.decoding)

                with open_dict(self.model.asr_model.cfg.decoding):
                    cfg = OmegaConf.merge(self.model.asr_model.cfg.decoding, self.cfg.rnnt_decoding)
                logger.debug(cfg)

                self.model.asr_model.change_decoding_strategy(OmegaConf.create(cfg))
                #self.model.asr_model.change_decoding_strategy(self.model.asr_model.cfg)
                #asr_model.change_decoding_strategy(asr_model.cfg.decoding)
        else:
            return

    @staticmethod
    def add_args(parser):
        parser.add_argument("--sfm_pretrained_name", type=str, default="nvidia/parakeet-tdt-0.6b-v3")
        parser.add_argument("--sfm_manifest_path", type=str, default="vp.jsonl")

        # Streaming / Windowing
        parser.add_argument("--sfm_chunk_secs", type=float, default=1)
        parser.add_argument("--sfm_left_context_secs", type=float, default=20)
        parser.add_argument("--sfm_right_context_secs", type=float, default=0)

        # ----------------------------------------------------------------
        # NEW: max_empty_chunks
        # When max_empty_chunks > 0, the LCP hyp_buffer is forced to clean the
        # n-1 buffer, skipping any words that may be blocking the commitment of
        # any new words.
        # ----------------------------------------------------------------
        parser.add_argument("--sfm_max_empty_chunks", type=int, default=0)

        # Emission Policies
        parser.add_argument("--sfm_policy", type=str, default="LACP",
                            choices=["LCP", "LACP", "SLCP", "WaitK", "HoldN"])
        parser.add_argument("--sfm_lacp_threshold", type=float, default=2)
        parser.add_argument("--sfm_K", type=int, default=2)
        parser.add_argument("--sfm_N", type=int, default=5)
        parser.add_argument("--sfm_emit_incomplete", type=bool, default=False,
                            help="Emit sub-word tokens at the end of a segment")

        # SLCP-specific arguments
        parser.add_argument("--sfm_slcp_semantic_threshold", type=float, default=0.65,
                            help="Morphological similarity threshold for SLCP Pass 3 "
                                 "(used when --sfm_slcp_use_spacy is not set). "
                                 "Values in [0, 1]; higher = stricter. E.g. 0.8.")
        parser.add_argument("--sfm_slcp_max_gap", type=int, default=3,
                            help="SLCP Pass 4: max consecutive unstable tokens that "
                                 "can be bridged when propagating the stable prefix.")
        parser.add_argument("--sfm_slcp_use_spacy", action="store_true",
                            help="Enable spacy-based linguistic checks in SLCP Pass 3 "
                                 "(requires en_core_web_sm to be installed).")

        # ----------------------------------------------------------------
        # NEW: word_level flag
        # When True, hypothesis buffers operate on complete words rather than
        # individual SentencePiece tokens.  The incomplete-word guard in the
        # agent policy loop is automatically bypassed (it becomes redundant).
        # ----------------------------------------------------------------
        parser.add_argument("--sfm_word_level", action="store_true",
                            help="Run emission policy at word level instead of token level")

        # Hardware
        parser.add_argument("--sfm_device", type=str, default="cuda")
        parser.add_argument("--sfm_compute_dtype", type=str, default="bfloat16",
                            choices=["float16", "float32", "bfloat16"])
        parser.add_argument("--sfm_decode", type=int, default=1)

        parser.add_argument("--sfm_context_score", type=float, default=1.0)
        parser.add_argument("--sfm_depth_scaling", type=float, default=2.0)
        parser.add_argument("--sfm_boosting_tree_alpha", type=float, default=1.0)
        parser.add_argument("--sfm_mbr", action="store_true")
        parser.add_argument("--pdfs", action="store_true")

    def build_states(self) -> ParakeetStreamingStates:
        do_debug = logger.level == logging.DEBUG
        word_level = getattr(self.cfg, 'word_level', False)

        audio_buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size=1,
            context_samples=self.model.context_samples,
            dtype=self.model.dtype,
            device=self.model.device,
        )

        if self.cfg.policy == 'LCP':
            hyp_buffer = LCPHypothesisBuffer(word_level=word_level, debug=do_debug)
        elif self.cfg.policy == 'LACP':
            hyp_buffer = LACPHypothesisBuffer(self.cfg.lacp_threshold, word_level=word_level, debug=do_debug)
        elif self.cfg.policy == 'SLCP':
            hyp_buffer = _build_slcp_buffer(self.cfg, word_level=word_level, debug=do_debug)
        elif self.cfg.policy == 'WaitK':
            hyp_buffer = WaitKHypothesisBuffer(
                self.cfg.K,
                features_per_second=self.model.features_per_sec,
                subsampling_factor=self.model.subsampling_factor,
                word_level=word_level,
                debug=do_debug,
            )
        else:
            hyp_buffer = HoldNHypothesisBuffer(self.cfg.N, word_level=word_level, debug=do_debug)

        return ParakeetStreamingStates(
            buffer=audio_buffer,
            hyp_buffer=hyp_buffer,
            current_offset=0,
            encoder_frame2audio_samples=self.model.encoder_frame2audio_samples,
            left_sample=0,
            right_sample=self.model.context_samples.chunk + self.model.context_samples.right,
            nchunks_no_output=0,
            incomplete_buffer=[],
            dtype=self.model.dtype,
            device=self.model.device,
            debug=do_debug,
        )

    def reset(self):
        self.states.reset(self.cfg, self.model)

    @torch.no_grad()
    def policy(self, states: Optional[AgentStates] = None):
        if not states.buffer.samples.shape[1]:
            logger.debug("[ASR] Empty states,bufffer.samples. Emting Read")
            return ReadAction()

        hyp = self.model.process_chunk(states.buffer, states.current_offset)

        states.hyp_buffer.insert(hyp)

        max_empty_chunks = getattr(self.cfg, 'max_empty_chunks', 0)
        if self.cfg.policy == 'WaitK':
            out = states.hyp_buffer.flush(
                last_instant=states.left_sample // states.encoder_frame2audio_samples
            )
        elif self.cfg.policy == 'LACP':
            out = states.hyp_buffer.flush(forced=True)
        elif self.cfg.policy == 'LCP' and states.nchunks_no_output >= max_empty_chunks:
            out = states.hyp_buffer.flush(forced=True)
        else:
            out = states.hyp_buffer.flush()

        logger.debug
        if states.source_finished:
            out.extend(states.hyp_buffer.complete())
            out_toks = [t for _, _, t in out]
            out_text = self.model.asr_model.tokenizer.tokens_to_text(out_toks) if out_toks else ""
            logger.debug(f"[ASR] Source finished, will emit {out_text=}")
            return WriteAction(out_text, finished=True)

        if max_empty_chunks:
            if not out:
                states.nchunks_no_output += 1
                if states.nchunks_no_output >= max_empty_chunks:
                    logger.info(f"ASR produced {max_empty_chunks} empty chunks. Forcing next commitment.")
            else:
                states.nchunks_no_output = 0

        if not out:
            logger.debug("[ASR] No output, emiting read")
            return ReadAction()

        out_toks = [t for _, _, t in out]

        # ------------------------------------------------------------------
        # Incomplete-word guard
        #
        # When word_level=True the hypothesis buffer already works on full
        # words, so this guard is entirely redundant and is skipped.
        #
        # When word_level=False (token-level) we filter out any trailing
        # sub-word continuation tokens that do not yet form a complete word.
        #
        # BUG FIX #6 (original): the original code stored out_toks[last_word:]
        # in incomplete_buffer, which kept the '▁' boundary token itself.
        # On the next call that token was prepended again, effectively doubling
        # the first character of the next word.
        # Fix: store out_toks[last_word + 1:] (everything *after* the last
        # boundary), and include out_toks[last_word] in the committed output.
        # ------------------------------------------------------------------
        word_level = getattr(self.cfg, 'word_level', False)

        if not word_level and not self.cfg.emit_incomplete:
            # Prepend any leftovers from the previous call
            #if states.incomplete_buffer:
            #    out_toks[:0] = states.incomplete_buffer
            #    states.incomplete_buffer = []

            #if not states.source_finished:
            #    # Find the last token that starts a new word
            #    last_word = -1
            #    for i in range(len(out_toks) - 1, -1, -1):
            #        if out_toks[i].startswith('▁'):
            #            last_word = i
            #            break

            #    if last_word != -1:
            #        # BUG FIX #6: store everything *after* the boundary token,
            #        # not *from* it.  The boundary token itself is complete.
            #        states.incomplete_buffer = out_toks[last_word + 1:]
            #        out_toks = out_toks[:last_word + 1]
            commit_toks = states.hyp_buffer.insert(out_toks)

        if word_level:
            # out_toks are already whole words (e.g. '▁briefly').
            # tokens_to_text expects individual SentencePiece sub-word pieces
            # and won't recognise concatenated words that aren't in the vocab,
            # printing the raw '▁' character instead of a space.
            # Simply replace '▁' → ' ' and strip the leading space.
            out_text = ''.join(t.replace('▁', ' ') for t in out_toks).strip()
        else:
            out_text = self.model.asr_model.tokenizer.tokens_to_text(out_toks)


        logger.info(f"ASR OUT {states.source_finished=} {out_text}")
        return WriteAction(out_text, finished=states.source_finished)
