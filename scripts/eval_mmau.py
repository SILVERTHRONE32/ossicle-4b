"""Run Ossicle on MMAU test-mini and write predictions in the format of the benchmark's own evaluation.py.
    python scripts/eval_mmau.py --checkpoint checkpoint --parquet test_mini.parquet --out out/mmau.json
    python evaluation.py --input out/mmau.json          # evaluation.py from the MMAU repository"""
import argparse, ast, json, re

import pyarrow.parquet as pq

from ossicle import Ossicle, load_audio
from ossicle.prompts import options_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

strip = lambda c: re.sub(r"^\(?[A-Da-d]\)\s*", "", str(c)).strip()
model = Ossicle.from_pretrained(a.checkpoint)
rows = pq.read_table(a.parquet).to_pylist()
rows = rows[: a.limit] if a.limit else rows
out = []
for r in rows:
    attrs = r["other_attributes"] if isinstance(r["other_attributes"], dict) else json.loads(r["other_attributes"])
    choices = r["choices"] if isinstance(r["choices"], list) else ast.literal_eval(r["choices"])
    mono, _ = load_audio(r["context"]["bytes"])
    reply = model.generate([("audio", mono, None)], options_prompt(r["instruction"], choices), max_new_tokens=60)
    out.append({"id": r.get("id"), "question": r["instruction"], "choices": [strip(c) for c in choices], "answer": strip(r["answer"]),
                "task": attrs["task"], "difficulty": attrs.get("difficulty") or "medium",
                "category": attrs.get("category"), "sub-category": attrs.get("sub-category") or attrs.get("sub_category"),
                "model_output": reply})
json.dump(out, open(a.out, "w"))
print("saved", len(out))
