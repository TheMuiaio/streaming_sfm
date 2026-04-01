from simuleval import simuleval
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.utils import entrypoint

from streaming_model import StreamingParakeet, StreamingBatchedAudioBufferWithOffset
from hyp_utils import (
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
    left_sample: int
    right_sample: int

    def reset(self):
        super().reset()
        self.buffer.reset_buffer() # Set to empty as in agent.build_states
        self.hyp_buffer.reset() # Set to empty as in agent.build_states
        self.current_offset = 0
        self.left_sample = 0
        self.right_sample = 0 # Set to right sample as in agent.build_states


@entrypoint
class ParakeetStreamingAgent(SpeechToTextAgent):
    def __init__(self, args: Namespace):
        super().__init__(args)

        # Mapping args to conf and setting cuda settings
        self.cfg = OmegaConf.create(vars(args))
        with open_dict(self.cfg):
            self.cfg.cuda = 0 if args.device == "cuda" else -1
            self.cfg.allow_mps = True if args.device == "mps" else False

        # Building the streaming model
        model_id = args.model_path or args.pretrained_name
        if not model_id:
            raise ValueError("Neither of --model_path or --pretrained_name were provided")
        else:
            print(f"--- Initializing Streaming Parakeet ---")
            self.model = StreamingParakeet(self,cfg)

    @staticmethod
    def add_args(parser: ArgumentParser):
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
            context_samples=self.model.context_samples,
            device=self.model.device,
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
            left_sample=0,
            right_sample=self.model.context_samples.chunk + self.model.context_samples.right
        )

    def reset(self):
        super().reset()

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            print("States is none. Setting to self.states")
            states = self.states
        new_samples_available = len(states.audio_source) - states.processed_samples
        
        if new_samples_available < self.chunk_samples and not states.finish_read:
            return ReadAction()

        start = states.processed_samples
        end = start + self.chunk_samples if not states.finish_read else len(states.audio_source)
        
        chunk = torch.tensor(states.audio_source[start:end], device=self.model.device).unsqueeze(0)
        chunk_len = torch.tensor([chunk.shape[1]], device=self.model.device)
        
        stride = states.audio_buffer.add_audio_batch_get_stride(
            chunk,
            audio_lengths=chunk_len,
            is_last_chunk=states.finish_read,
            is_last_chunk_batch=torch.tensor([states.finish_read], device=self.model.device)
        )
        
        # Update offsets for timestamp alignment
        states.current_offset += (stride // self.model.encoder_frame2audio_samples)
        states.processed_samples = end

        formatted_hyp = self.model.process_chunk(states.audio_buffer, states.current_offset)
        
        states.hyp_buffer.insert(formatted_hyp)
        
        if self.model.cfg.policy == 'WaitK':
            # Calculate the current 'instant' for Wait-K
            last_instant = (states.processed_samples // self.model.encoder_frame2audio_samples)
            tokens_to_emit = states.hyp_buffer.flush(last_instant=last_instant)
        else:
            tokens_to_emit = states.hyp_buffer.flush()

        if states.finish_read:
            tokens_to_emit.extend(states.hyp_buffer.complete())

        if tokens_to_emit:
            # Join emitted tokens into a string
            prediction_text = " ".join([t for _, _, t in tokens_to_emit])
            return WriteAction(prediction_text, finished=states.finish_read)

        # If policy didn't emit anything, read more audio
        return ReadAction()