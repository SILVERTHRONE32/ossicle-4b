"""Step 2. Calibration inputs for the quantizer: full input-embedding sequences (prefix + audio + question) built from your own clips.
    python nvfp4/calibration.py --checkpoint checkpoint --data calib.jsonl --out calibration.pt
`calib.jsonl`: one {"audio": "path.wav", "question": "..."} per line; a few hundred lines covering the kinds of audio and questions you expect (speech, sound, music)."""
import argparse
import json

import torch

from ossicle import Ossicle, load_audio

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = Ossicle.from_pretrained(a.checkpoint)
seqs = []
for line in open(a.data):
    it = json.loads(line)
    mono, _ = load_audio(it["audio"], 30)
    with torch.no_grad():
        x, _, _, _ = m.build([{"segments": [("audio", mono, None)], "question": it["question"]}])
    seqs.append(x[0].cpu())
torch.save(seqs, a.out)
print(len(seqs), "sequences")
