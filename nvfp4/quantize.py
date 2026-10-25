"""Step 3. Quantize the language model to NVFP4 with NVIDIA ModelOpt (standard export, loadable by vLLM). Needs `pip install nvidia-modelopt[hf]`.
    python nvfp4/quantize.py --lm lm_bf16 --calibration calibration.pt --out lm_nvfp4"""
import argparse

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--lm", required=True)
ap.add_argument("--calibration", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.lm, dtype=torch.bfloat16).cuda().eval()
calib = torch.load(a.calibration)


def loop(m):
    with torch.no_grad():
        for x in calib:
            m(inputs_embeds=x[None].to("cuda", torch.bfloat16))


model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, loop)
model.config.architectures = ["Qwen3_5ForCausalLM"]
export_hf_checkpoint(model, export_dir=a.out)
AutoTokenizer.from_pretrained(a.lm).save_pretrained(a.out)
