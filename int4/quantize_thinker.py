"""Qwen3.5-4B (the thinking model of the two-pass system) to INT4 (W4A16, groups of 128) with GPTQ, in the compressed-tensors format vLLM loads.
    python int4/quantize_thinker.py --model Qwen/Qwen3.5-4B --calib calib_text.json --out thinker_int4
calib_text.json is the same file nvfp4/quantize_thinker.py reads."""
import argparse, json, random

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
random.seed(0)
tok = AutoTokenizer.from_pretrained(a.model)
C = json.load(open(a.calib))
texts = []
for cap in C["caps"][:256]:
    q = random.choice(["What is going on in this clip?", "What instruments are playing?", "Why do you think so?", "What is the mood?", "Describe it in more detail."])
    texts.append(f"<|im_start|>user\nYou are talking with a user about an audio recording. WHAT YOU HEARD: {cap}\n\nUSER: {q}<|im_end|>\n<|im_start|>assistant\n<think>\n")
for x in C["qa"][:192]:
    texts.append("<|im_start|>user\n" + x["question"] + "\nOptions:\n" + "\n".join("- " + o for o in x["options"]) + "<|im_end|>\n<|im_start|>assistant\n<think>\n" + x["answer"])
ds = Dataset.from_dict({"text": texts}).map(lambda s: tok(s["text"], max_length=512, truncation=True, add_special_tokens=False), remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*linear_attn.in_proj_a", "re:.*linear_attn.in_proj_b"], dampening_frac=0.01)
oneshot(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=512, num_calibration_samples=len(texts))
model.config.architectures = ["Qwen3_5ForCausalLM"]
model.save_pretrained(a.out, save_compressed=True)
tok.save_pretrained(a.out)
print("wrote", a.out)
