"""Step 4. Assemble the single vLLM checkpoint: audio encoder + Ossicle's head (as the encoder's output layers) in bf16, the stereo adapter, plus the quantized language model (NVFP4 from nvfp4/quantize.py or INT4 from int4/quantize.py).
    python nvfp4/assemble.py --checkpoint checkpoint --lm lm_nvfp4 --out Ossicle-4B-NVFP4
Needs TOWER_PATH (the Qwen3-Omni audio encoder) and LLM_PATH (Qwen3.5-4B) in the environment, as for the rest of the code."""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--lm", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tower = os.environ["TOWER_PATH"]
llm_cfg = json.load(open(os.path.join(os.environ["LLM_PATH"], "config.json")))["text_config"]
audio_cfg = json.load(open(f"{tower}/config.json"))
for d in (audio_cfg, llm_cfg):
    d.pop("transformers_version", None)
audio_cfg["output_dim"] = llm_cfg["hidden_size"]

tensors = {}
with safe_open(glob.glob(f"{tower}/*.safetensors")[0], "pt") as f:
    for k in f.keys():
        if not k.startswith(("proj1", "proj2")) and "positional_embedding" not in k:
            tensors["audio_tower." + k] = f.get_tensor(k).to(torch.bfloat16).contiguous()
for k, v in torch.load(f"{a.checkpoint}/head.pt", map_location="cpu").items():            # the trained head replaces the tower's own output layers
    tensors["audio_tower." + k] = v.to(torch.bfloat16).contiguous()
has_stereo = os.path.exists(f"{a.checkpoint}/stereo_adapter.pt")
if has_stereo:                                                                               # spatial hearing: kept in 32-bit, applied inside vLLM
    for k, v in torch.load(f"{a.checkpoint}/stereo_adapter.pt", map_location="cpu").items():
        tensors["stereo_adapter." + k] = v.float().contiguous()
with safe_open(f"{a.lm}/model.safetensors", "pt") as f:
    for k in f.keys():
        tensors["language_model." + k] = f.get_tensor(k)
os.makedirs(a.out, exist_ok=True)
save_file(tensors, f"{a.out}/model.safetensors", metadata={"format": "pt"})

# the layers that stay in 16-bit (linear-attention gates and convolutions, embeddings, lm_head), named as vLLM sees them
ignore = ["language_model.lm_head", "language_model.model.embed_tokens", "lm_head", "model.language_model.embed_tokens"]
for i, t in enumerate(llm_cfg["layer_types"]):
    if t == "linear_attention":
        for m in ("in_proj_a", "in_proj_b", "conv1d"):
            ignore += [f"language_model.model.layers.{i}.linear_attn.{m}", f"model.language_model.layers.{i}.linear_attn.{m}"]
quant = json.load(open(f"{a.lm}/config.json"))["quantization_config"]
quant["ignore"] = ignore
json.dump({"model_type": "ossicle", "architectures": ["OssicleForConditionalGeneration"], "audio_cfg": audio_cfg, "llm_cfg": llm_cfg, "has_stereo": has_stereo,
           "dtype": "bfloat16", "quantization_config": quant}, open(f"{a.out}/config.json", "w"), indent=2)
if os.path.exists(f"{a.lm}/hf_quant_config.json"):                                           # NVIDIA ModelOpt (NVFP4); INT4 compressed-tensors keeps everything in config.json
    hq = json.load(open(f"{a.lm}/hf_quant_config.json"))
    hq["quantization"]["exclude_modules"] = ignore
    json.dump(hq, open(f"{a.out}/hf_quant_config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if os.path.exists(f"{a.lm}/{f}"):
        shutil.copy(f"{a.lm}/{f}", a.out)
for f in ("merges.txt", "vocab.json", "preprocessor_config.json"):
    for src in (a.lm, tower, os.environ["LLM_PATH"]):
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", a.out)
            break
shutil.copy(os.path.join(os.path.dirname(__file__), "chat_template.jinja"), a.out)
print("wrote", a.out)
