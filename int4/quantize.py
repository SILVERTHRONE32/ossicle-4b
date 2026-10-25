"""INT4 (W4A16, groups of 128, symmetric, round to nearest) in the compressed-tensors format that vLLM loads. Needs `pip install llmcompressor`.
Works on any standard Qwen3.5 checkpoint, for example the merged Ossicle language model (nvfp4/merge_lm.py) or the Qwen3.5-4B thinking model.
    python int4/quantize.py --model lm_bf16 --out lm_int4"""
import argparse

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
recipe = QuantizationModifier(targets="Linear", scheme="W4A16",
                              ignore=["lm_head", "re:.*linear_attn.in_proj_a", "re:.*linear_attn.in_proj_b", "re:.*linear_attn.conv1d"])
oneshot(model=model, recipe=recipe)
model.config.architectures = ["Qwen3_5ForCausalLM"]
model.save_pretrained(a.out, save_compressed=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print("wrote", a.out)
