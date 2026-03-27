from simuleval import simuleval
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import ReadAction, WriteAction

from streaming_model import StreamingParakeet, StreamingBatchedAudioBufferWithOffset
from hyp_utils import ABSHypothesisBuffer

from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass
from argparse import Namespace, ArgumentParser

@dataclass
class ParakeetStreamingStates(AgentStates):
    buffer: StreamingBatchedAudioBufferWithOffset
    hyp_buffer: ABSHypothesisBuffer
    current_offset: int
    left_sample: int
    right_sample: int

    def reset(self):
        super().reset()
        self.buffer = StreamingBatchedAudioBufferWithOffset() # Set to empty as in agent.build_states
        self.hyp_buffer = ABSHypothesisBuffer() # Set to empty as in agent.build_states
        self.current_offset = 0
        self.left_sample = 0
        self.right_sample = 0 # Set to right sample as in agent.build_states


class ParakeetStreamingAgent(SpeechToTextAgent):
    def __init__(self, args: Namespace):
        super().__init__(args)

        # Mapping args to conf and setting cuda settings
        cfg = OmegaConf.create(vars(args))
        with open_dict(cfg):
            cfg.cuda = 0 if args.device == "cuda" else 0
            cfg.allow_mps = True if args.device == "mps" else False

        # Building the streaming model
        model_id = args.model_path or args.pretrained_name
        if not model_id:
            raise ValueError("Neither of --model_path or --pretrained_name were provided")
        else:
            print(f"--- Initializing Streaming Parakeet ---")
            self.model = StreamingParakeet(cfg)

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
        return ParakeetStreamingStates()

    def policy(self):
        pass