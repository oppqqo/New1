from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs, TransformerModel

class LLaMA:
    def __init__(self, model: TransformerModel, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
    
    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path:str, load_model:bool, max_seq_len:int, max_batch_size:int, device:str) -> 'LLaMA':
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoints found in {checkpoint_dir}"
            ckpt_path = checkpoints[0]
            print(f"loading checkpoint {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"loading checkpoint takes {(time.time() - prev_time):.2f} seconds")
            prev_time = time.time()
        with open(Path(checkpoint_dir) / "params.json", "r") as f:
            params = json.load(f)
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"loading model takes {(time.time() - prev_time):.2f} seconds")
        return LLaMA(model, tokenizer, model_args)
if __name__ == "__main__":
    torch.manual_seed(0)
    allow_cuda = False
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"
    model = LLaMA.build(
        checkpoint_dir="checkpoints/llama-2-7b-chat",
        tokenizer_path="checkpoints/llama-2-7b-chat/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device
    )
    print("All done!")
