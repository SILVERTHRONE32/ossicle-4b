"""Fine-tune the head and LoRA adapters on audio question-answer pairs.
    python scripts/train.py --init checkpoint --data train.jsonl --out runs/mine
`train.jsonl`: one JSON object per line, {"audio": "path.wav", "question": "...", "answer": "..."}."""
import argparse, json, math, os, random, time

import torch

from ossicle import Ossicle, load_audio

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--head_lr", type=float, default=2e-5)
ap.add_argument("--max_seconds", type=float, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

items = [json.loads(l) for l in open(a.data)]
model = Ossicle.from_pretrained(a.init, trainable=True)
lora = [p for p in model.llm.parameters() if p.requires_grad]
for p in lora:
    p.data = p.data.float()
model.llm.gradient_checkpointing_enable()
head = list(model.head.parameters())
opt = torch.optim.AdamW([{"params": head, "lr": a.head_lr}, {"params": lora, "lr": a.lr}], weight_decay=0.0, betas=(0.9, 0.98))
base = [g["lr"] for g in opt.param_groups]
rng = random.Random(a.seed)
t0 = time.time()
for step in range(1, a.steps + 1):
    scale = max(min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps)), 0.02)
    for g, b in zip(opt.param_groups, base):
        g["lr"] = b * scale
    batch = []
    for it in rng.sample(items, a.batch_size):
        mono, _ = load_audio(it["audio"], a.max_seconds)
        batch.append({"segments": [("audio", mono, None)], "question": it["question"], "answer": it["answer"]})
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head + lora, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 20 == 0 or step == 1:
        print(json.dumps({"step": step, "loss": round(loss.item(), 3), "s_per_step": round((time.time() - t0) / step, 2)}), flush=True)
model.save_pretrained(a.out)
print("saved", a.out)
