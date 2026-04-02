from simuleval.agents import SpeechToTextAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.utils import entrypoint

from streaming_sfm.streaming_model import StreamingParakeet, StreamingBatchedAudioBufferWithOffset
from streaming_sfm.hyp_utils import (
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

@dataclass
class ParakeetStreamingStates(AgentStates):
    buffer: StreamingBatchedAudioBufferWithOffset
    hyp_buffer: ABSHypothesisBuffer
    current_offset: int
    encoder_frame2audio_samples: int
    left_sample: int
    right_sample: int
    dtype: torch.dtype
    device: torch.device

    def reset(self, cfg, asr_model): 
        super().reset()

        self.buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size = 1,
            context_samples = asr_model.context_samples,
            dtype = asr_model.dtype,
            device = asr_model.device,
        )

        self.hyp_buffer = self.reset_hyp_buffer(cfg, asr_model)
        self.current_offset = 0
        self.encoder_frame2audio_samples = asr_model.encoder_frame2audio_samples
        self.left_sample = 0
        self.right_sample = asr_model.context_samples.chunk + asr_model.context_samples.right
        self.dtype = asr_model.dtype
        self.device = asr_model.device

    def reset_hyp_buffer(self, cfg, model):
        if cfg.policy == 'LCP':
            hyp_buffer = LCPHypothesisBuffer()
        elif cfg.policy == 'LACP':
            hyp_buffer = LACPHypothesisBuffer(cfg.lacp_threshold)
        elif cfg.policy == 'WaitK':
            hyp_buffer = WaitKHypothesisBuffer(
                cfg.K,
                features_per_second=model.features_per_sec,
                subsampling_factor=model.subsampling_factor,
            )
        else:
            hyp_buffer = HoldNHypothesisBuffer(cfg.N)
        return hyp_buffer

    def update_source(self, segment):
        stride = self.buffer.add_audio_batch_get_stride(
            torch.tensor([segment.content], device = self.device),
            audio_lengths=torch.tensor([len(segment.content)], device = self.device),
            is_last_chunk=segment.finished,
            is_last_chunk_batch=torch.tensor([segment.finished], device = self.device),
            )

        self.current_offset += (stride // self.encoder_frame2audio_samples)
        super().update_source(segment)

@entrypoint
class ParakeetAgent(SpeechToTextAgent):
    def __init__(self, args: Namespace):
        self.cfg = OmegaConf.create(vars(args))
        with open_dict(self.cfg):
            self.cfg.cuda = 0 if args.device == "cuda" else -1
            self.cfg.allow_mps = True if args.device == "mps" else False

        model_id = args.model_path or args.pretrained_name
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
        if not model_id:
            raise ValueError("Neither of --model_path or --pretrained_name were provided")
        print(f"--- Initializing Streaming Parakeet ---")
        self.model = StreamingParakeet(self.cfg)
        super().__init__(args)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--model_path", type=str, default=None, help="Path to .nemo file")
        parser.add_argument("--pretrained_name", type=str, default=None, help="Name of a pretrained model")
        parser.add_argument("--manifest_path", type=str, default="vp.jsonl", help="Path to NeMo manifest")
    
        # Streaming / Windowing
        parser.add_argument("--chunk_secs", type=float, default=1, help="Duration of the sliding window chunk")
        parser.add_argument("--left_context_secs", type=float, default=20, help="Left context duration")
        parser.add_argument("--right_context_secs", type=float, default=0, help="Right context duration")
    
        # Emission Policies
        parser.add_argument("--policy", type=str, default="LACP", choices=["LCP", "LACP", "WaitK", "HoldN"])
        parser.add_argument("--lacp_threshold", type=float, default=2, help="Threshold for LACP policy")
        parser.add_argument("--K", type=int, default=2, help="K value for WaitK policy")
        parser.add_argument("--N", type=int, default=5, help="N value for HoldN policy")
    
        # Hardware
        parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
        parser.add_argument("--compute_dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])

    def build_states(self) -> ParakeetStreamingStates:
        audio_buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size = 1,
            context_samples = self.model.context_samples,
            dtype = self.model.dtype,
            device = self.model.device,
        )

        if self.cfg.policy == 'LCP':
            hyp_buffer = LCPHypothesisBuffer()
        elif self.cfg.policy == 'LACP':
            hyp_buffer = LACPHypothesisBuffer(self.cfg.lacp_threshold)
        elif self.cfg.policy == 'WaitK':
            hyp_buffer = WaitKHypothesisBuffer(
                self.cfg.K,
                features_per_second=self.model.features_per_sec,
                subsampling_factor=self.model.subsampling_factor,
            )
        else:
            hyp_buffer = HoldNHypothesisBuffer(self.cfg.N)
        return ParakeetStreamingStates(
            buffer=audio_buffer,
            hyp_buffer=hyp_buffer,
            current_offset=0,
            encoder_frame2audio_samples=self.model.encoder_frame2audio_samples,
            left_sample=0,
            right_sample=self.model.context_samples.chunk + self.model.context_samples.right,
            dtype=self.model.dtype,
            device=self.model.device,
        )

    def reset(self):
        self.states.reset(self.cfg, self.model)
    
    @torch.no_grad()
    def policy(self, states: Optional[AgentStates] = None):
        if not states.buffer.samples.shape[1]:
            return ReadAction()
        
        hyp = self.model.process_chunk(states.buffer, states.current_offset)

        hyp_buffer = states.hyp_buffer.insert(hyp)

        if self.cfg.policy == 'WaitK':
            out = states.hyp_buffer.flush(last_instant=states.left_sample // states.encoder_frame2audio_samples)
        else:
            out = states.hyp_buffer.flush()
        
        if states.source_finished:
            out.extend(states.hyp_buffer.complete())

        if not out:
            # Empty token buffer. Return ReadAction
            return ReadAction()
        
        ## Gestionar paraules incompletes !!!!!!!!!
        #out_text = ' '.join([t for _, _, t in out])
        out_toks = [t for _, _, t in out]
        out_text = self.model.asr_model.tokenizer.tokens_to_text(out_toks)
        return WriteAction(out_text, finished=states.source_finished)
