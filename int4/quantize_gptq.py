"""INT4 (W4A16, groups of 128) with GPTQ for the Ossicle language model, calibrated on audio-embedding sequences (nvfp4/calibration.py), in the compressed-tensors format vLLM loads.
    python int4/quantize_gptq.py --lm lm_bf16 --calibration calibration.pt --out lm_int4"""
import argparse

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--lm", required=True)
ap.add_argument("--calibration", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.lm)
model = AutoModelForCausalLM.from_pretrained(a.lm, dtype=torch.bfloat16)
seqs = torch.load(a.calibration)
loader = DataLoader(seqs, batch_size=1, collate_fn=lambda b: {"inputs_embeds": b[0][None].to(torch.bfloat16)})
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*linear_attn.in_proj_a", "re:.*linear_attn.in_proj_b"], dampening_frac=0.01)
oneshot(model=model, processor=tok, dataset=loader, recipe=recipe, num_calibration_samples=len(seqs))
model.config.architectures = ["Qwen3_5ForCausalLM"]
model.save_pretrained(a.out, save_compressed=True)
tok.save_pretrained(a.out)
print("wrote", a.out)
