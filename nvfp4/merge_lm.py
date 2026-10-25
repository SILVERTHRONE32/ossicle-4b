"""Step 1. Fold the LoRA into Qwen3.5-4B and save the language model as a standard bf16 Qwen3_5ForCausalLM checkpoint.
    python nvfp4/merge_lm.py --checkpoint checkpoint --out lm_bf16"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ossicle import Ossicle
from ossicle.model import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = Ossicle.from_pretrained(a.checkpoint, device="cpu")
m.llm = m.llm.merge_and_unload()
lm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16)
print(lm.load_state_dict({"model." + k: v for k, v in m.llm.state_dict().items()}, strict=False))      # only the tied lm_head is missing
lm.tie_weights()
lm.save_pretrained(a.out)
AutoTokenizer.from_pretrained(LLM).save_pretrained(a.out)
