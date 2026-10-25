"""Longest audio that fits on this GPU: peak GPU memory by audio length, searched by doubling and then bisecting, each length in a fresh process.
The clip is tiled to the requested length (a memory test, not a comprehension test); the question is answered with 60 new tokens.
    python scripts/bench_long.py checkpoint clip.wav --quant bf16 --mode long
    python scripts/bench_long.py checkpoint clip.wav --quant int4 --mode whole      # the whole clip through the encoder at once, for comparison
    python scripts/bench_long.py checkpoint clip.wav --quant bf16 --clip_seconds 30 --start 10 --resolution 5    # how many 30 s clips fit in one question
--quant int4: group-wise 4-bit weights (ossicle/w4.py) on the language model and the audio encoder; bf16 keeps the checkpoint as shipped."""
import argparse
import json
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("clip")
ap.add_argument("--quant", choices=["bf16", "int4"], default="bf16")
ap.add_argument("--mode", choices=["long", "whole"], default="long")
ap.add_argument("--clip_seconds", type=float, default=None, help="count mode: the question holds N clips of this length (cut in turn from the source clip) and N is searched instead of minutes")
ap.add_argument("--start", type=float, default=10, help="first length to try, in minutes (clips with --clip_seconds)")
ap.add_argument("--max_minutes", type=float, default=600, help="upper limit of the search (clips with --clip_seconds)")
ap.add_argument("--resolution", type=float, default=5, help="bisect until the boundary is known to this many minutes (clips with --clip_seconds)")
ap.add_argument("--one", type=float, default=None, help="internal: run a single length and print one JSON line")
a = ap.parse_args()


def one(minutes):
    import numpy as np
    import torch
    import ossicle.model as om
    from ossicle import Ossicle, load_audio
    from ossicle.w4 import quantize_linears
    om.LONG_AFTER = 120 if a.mode == "long" else 10 ** 9
    m = Ossicle.from_pretrained(a.checkpoint)
    if a.quant == "int4":
        m.llm = m.llm.merge_and_unload()
        quantize_linears(m.llm, skip=("embed",))
        quantize_linears(m.encoder)
    m.llm.eval()
    torch.cuda.empty_cache()
    weights = torch.cuda.memory_allocated() / 1e9
    mono = load_audio(a.clip)[0]
    if a.clip_seconds:
        n, L = int(minutes), int(a.clip_seconds * 16000)
        segments = []
        for i in range(n):
            s0 = (i * L) % max(len(mono) - L, 1)
            segments += [("text", ("\n" if i else "") + f"Audio {i + 1}:\n"), ("audio", mono[s0: s0 + L], None)]
        segments.append(("text", "\n"))
    else:
        secs = int(minutes * 60)
        segments = [("audio", np.tile(mono, -(-secs * 16000 // len(mono)))[: secs * 16000], None)]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        out = m.generate(segments, "Describe the audio in two sentences." if not a.clip_seconds else "Which recording is loudest?", 60)
        torch.cuda.synchronize()
        r = {"ok": not out.strip().startswith("!!!!"), "weights_gb": weights, "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
             "seconds": time.time() - t, "answer": out[:120]}
        if not r["ok"]:
            r["error"] = "degenerate output (NaN logits)"
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        r = {"ok": False, "error": "out of memory", "weights_gb": weights}
    print("RESULT " + json.dumps(r), flush=True)


def run(minutes):
    cmd = [sys.executable, __file__, a.checkpoint, a.clip, "--quant", a.quant, "--mode", a.mode, "--one", str(minutes)] + (["--clip_seconds", str(a.clip_seconds)] if a.clip_seconds else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r = json.loads(line[-1][7:]) if line else {"ok": False, "error": "crashed: " + p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else "crashed"}
    print(f"[{a.quant}/{a.mode}] {minutes:7.1f} {'clips' if a.clip_seconds else 'min'}: " + (f"ok, peak {r['peak_gb']:.2f} GB, {r['seconds']:.0f} s" if r["ok"] else r["error"]), flush=True)
    return r


if a.one is not None:
    one(a.one)
    sys.exit()
good, bad, best, n = 0, None, None, a.start
while n <= a.max_minutes:
    r = run(n)
    if r["ok"]:
        good, best, n = n, r, n * 2
    else:
        bad = n
        break
if bad is None and n > a.max_minutes and good < a.max_minutes:
    r = run(a.max_minutes)
    if r["ok"]:
        good, best = a.max_minutes, r
    else:
        bad = a.max_minutes
while bad is not None and bad - good > a.resolution:
    mid = round((good + bad) / 2 / a.resolution) * a.resolution
    mid = min(max(mid, good + a.resolution), bad - a.resolution) if bad - good > a.resolution else mid
    r = run(mid)
    if r["ok"]:
        good, best = mid, r
    else:
        bad = mid
print("SUMMARY " + json.dumps({"quant": a.quant, "mode": a.mode, "max_ok": good, "first_failure": bad, "unit": "clips" if a.clip_seconds else "minutes", "clip_seconds": a.clip_seconds, "peak_gb_at_max": best and best["peak_gb"], "weights_gb": best and best["weights_gb"], "seconds_at_max": best and best["seconds"]}), flush=True)
