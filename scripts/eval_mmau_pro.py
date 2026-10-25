"""Run Ossicle on MMAU-Pro and write a parquet for the authors' scorer.
    python scripts/eval_mmau_pro.py --checkpoint checkpoint --data /path/to/mmau-pro --out out/single.json
`--data` holds test.parquet and data.zip. Resumable; --shard i/n splits the questions."""
import argparse, io, json, os, re, time, zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from ossicle import Ossicle, load_audio
from ossicle.prompts import letter_prompt, letter_to_option, options_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", choices=["options", "letters"], default="options")
ap.add_argument("--closed_only", action="store_true")
ap.add_argument("--shard", default="")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

OPEN = ("open", "instruction following")
table = pq.read_table(f"{a.data}/test.parquet")
rows = table.to_pylist()
rows = rows[: a.limit] if a.limit else rows
rows = [r for r in rows if r["category"] not in OPEN] if a.closed_only else rows
if a.shard:
    i, n = map(int, a.shard.split("/"))
    rows = rows[i::n]

model = Ossicle.from_pretrained(a.checkpoint)
zf = zipfile.ZipFile(f"{a.data}/data.zip")
files = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist() if n.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))}
part = a.out + ".partial"
res = {k: v for k, v in (json.load(open(part)) if os.path.exists(part) else {}).items() if v}
t0 = time.time()
for r in rows:
    if r["id"] in res:
        continue
    cat, q = r["category"], r["question"]
    clips = [load_audio(zf.read(files[os.path.splitext(os.path.basename(p))[0]])) for p in r["audio_path"]]
    choices = [str(c) for c in r["choices"]] if cat not in OPEN else None
    if cat in OPEN:
        question = q
    else:
        question = (letter_prompt if a.prompt == "letters" else options_prompt)(q, choices)
    if len(clips) > 1:
        head = ["URL%d: " % (i + 1) if "URL" in q else "Audio %d:\n" % (i + 1) for i in range(len(clips))]
        segs = []
        for i, c in enumerate(clips):
            segs += [("text", head[i] if i == 0 else "\n" + head[i]), ("audio", c[0], None)]
        segs.append(("text", "\n"))
    else:
        mono, stereo = clips[0]
        segs = [("audio", mono, stereo if cat == "spatial_audio" else None)]
    out = model.generate(segs, question, max_new_tokens=200 if cat == "open" else 300 if cat == "instruction following" else 60)
    if a.prompt == "letters" and cat not in OPEN:
        out = letter_to_option(out, choices)
    res[r["id"]] = out
    if len(res) % 100 == 0:
        print(len(res), len(rows), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(part, "w"))
json.dump(res, open(a.out, "w"))
outs = [res.get(r["id"], "") for r in rows]
pq.write_table(table.slice(0, len(rows)).append_column("model_output", pa.array(outs, pa.string())), a.out.replace(".json", ".parquet"))
print("saved", len(res))
