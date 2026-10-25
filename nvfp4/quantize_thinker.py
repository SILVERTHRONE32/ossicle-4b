"""Qwen3.5-4B (the thinking model of the two-pass system) to NVFP4 with NVIDIA ModelOpt, in the standard export format vLLM loads.
    python nvfp4/quantize_thinker.py --model Qwen/Qwen3.5-4B --calib calib_text.json --out thinker_nvfp4
calib_text.json: {"caps": [clip descriptions], "qa": [{"question": ..., "options": [...], "answer": ...}]}, written from your own training pools."""
import argparse, json, random

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
random.seed(0)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
C = json.load(open(a.calib))
texts = []
for cap in C["caps"][:256]:  # conversation-shaped prompts, like the thinking model sees them
    q = random.choice(["What is going on in this clip?", "What instruments are playing?", "Why do you think so?", "What is the mood?", "Describe it in more detail."])
    texts.append(f"<|im_start|>user\nYou are talking with a user about an audio recording. WHAT YOU HEARD: {cap}\n\nUSER: {q}<|im_end|>\n<|im_start|>assistant\n<think>\n")
for x in C["qa"][:192]:
    texts.append("<|im_start|>user\n" + x["question"] + "\nOptions:\n" + "\n".join("- " + o for o in x["options"]) + "<|im_end|>\n<|im_start|>assistant\n<think>\n" + x["answer"])


def loop(m):
    with torch.no_grad():
        for t in texts:
            m(**tok(t, return_tensors="pt", truncation=True, max_length=768).to("cuda"))


model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, loop)
model.config.architectures = ["Qwen3_5ForCausalLM"]
export_hf_checkpoint(model, export_dir=a.out)
tok.save_pretrained(a.out)
print("wrote", a.out)
