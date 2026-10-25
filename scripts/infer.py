"""Ask a question about one or more audio files.
    python scripts/infer.py checkpoint clip.wav "What instruments can you hear?"
"""
import argparse

from ossicle import Ossicle

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("audio", nargs="+")
ap.add_argument("question")
ap.add_argument("--max_new_tokens", type=int, default=200)
a = ap.parse_args()
model = Ossicle.from_pretrained(a.checkpoint)
print(model.ask(a.audio if len(a.audio) > 1 else a.audio[0], a.question, a.max_new_tokens))
