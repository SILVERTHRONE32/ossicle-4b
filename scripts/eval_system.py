"""Two-pass system on MMAU-Pro: replaces the single-model answers of selected categories by answers written from
Ossicle's transcript, description and own answer. Input: the parquet/json from eval_mmau_pro.py (options prompt).
    python scripts/eval_system.py --checkpoint checkpoint --data /path/to/mmau-pro --single out/single.json --letters out/letters.json --out out/system.json
Speech and voice chat use the majority of --samples sampled answers."""
import argparse, collections, json, os, re, zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ossicle import Ossicle, load_audio
from ossicle.system import Reader, listen, prompt

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--single", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--letters", help="eval_mmau_pro.py output with --prompt letters, used for sound, music, spatial and sound+music")
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--limit", type=int, default=0, help="per category, for a quick test")
a = ap.parse_args()

KIND = {"voice_chat": "spoken_question", "instruction following": "spoken_instruction", "speech": "with_draft",
        "sound_speech": "with_draft", "open": "with_draft"}
VOTED = {"voice_chat", "speech"}
table = pq.read_table(f"{a.data}/test.parquet")
rows = [r for r in table.to_pylist() if r["category"] in KIND]
if a.limit:
    seen = collections.Counter()
    rows = [r for r in rows if seen.update([r["category"]]) or seen[r["category"]] <= a.limit]
single = json.load(open(a.single))
zf = zipfile.ZipFile(f"{a.data}/data.zip")
files = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist() if n.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))}

heard_path = a.out + ".heard"
heard = json.load(open(heard_path)) if os.path.exists(heard_path) else {}
model = Ossicle.from_pretrained(a.checkpoint)
for r in rows:
    if r["id"] not in heard:
        clips = [load_audio(zf.read(files[os.path.splitext(os.path.basename(p))[0]]), 120)[0] for p in r["audio_path"]]
        mono = clips[0] if len(clips) == 1 else np.concatenate([np.concatenate([c, np.zeros(8000, np.float32)]) for c in clips])
        heard[r["id"]] = dict(zip(("transcript", "description"), listen(model, mono)))
        if len(heard) % 25 == 0:
            json.dump(heard, open(heard_path, "w"))
json.dump(heard, open(heard_path, "w"))
del model
torch.cuda.empty_cache()

reader = Reader()
tok = lambda s: re.findall(r"[a-z0-9']+", str(s).lower())
def f1(x, y):
    cx, cy = collections.Counter(tok(x)), collections.Counter(tok(y)); o = sum((cx & cy).values())
    return 0 if o == 0 else 2 * o / (sum(cx.values()) + sum(cy.values()))
answers = {}
for k in range(a.samples):
    torch.manual_seed(k)
    for cat in KIND:
        batch = [r for r in rows if r["category"] == cat and (k == 0 or cat in VOTED)]
        for i in range(0, len(batch), 8):
            b = batch[i:i + 8]
            ps = [prompt(KIND[cat], heard[r["id"]]["transcript"], heard[r["id"]]["description"], r["question"],
                         None if cat in ("open", "instruction following") else [str(c) for c in r["choices"]],
                         single.get(r["id"])) for r in b]
            for r, o in zip(b, reader.answer(ps, answer_budget=250 if cat == "open" else 120)):
                answers.setdefault(r["id"], []).append(o)
    print("sample", k + 1, "done", flush=True)
final = {}
for r in rows:
    outs = answers[r["id"]]
    if r["category"] in VOTED and r["category"] != "open":
        ch = [str(c) for c in r["choices"]]
        idx = [max(range(len(ch)), key=lambda i: (f1(o, ch[i]), -i)) for o in outs]
        cnt = collections.Counter(idx); top = max(cnt.values()); win = [i for i, v in cnt.items() if v == top]
        final[r["id"]] = ch[idx[0] if idx[0] in win else win[0]]
    else:
        final[r["id"]] = outs[0]
base = pq.read_table(a.single.replace(".json", ".parquet"))
LETTER_CATS = {"sound", "music", "spatial_audio", "sound_music"}
letters = json.load(open(a.letters)) if a.letters else {}
out = [final.get(r["id"], letters.get(r["id"]) if r["category"] in LETTER_CATS else None) or r["model_output"] for r in base.to_pylist()]
i = base.schema.get_field_index("model_output")
pq.write_table(base.set_column(i, "model_output", pa.array(out, pa.string())), a.out.replace(".json", ".parquet"))
json.dump(final, open(a.out, "w"))
print("replaced", len(final))
