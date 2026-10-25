# Audio-Graft-4B

A 4.9B audio-language model that answers questions about speech, sound and music. It pairs the Qwen3-Omni audio encoder with Qwen3.5-4B and runs on a single 16 GB GPU.

## Model Overview

- **Type:** Audio-language model (audio + text in, text out)
- **Parameters:** 4.9B in total (audio encoder 0.65B, language model 4.2B); 27M trained
- **Audio encoder:** Qwen3-Omni AuT, frozen. Hidden size 1,280; 32 layers; 12.5 audio tokens per second
- **Language model:** Qwen3.5-4B with LoRA adapters
  - Hidden Dimension: 2,560
  - Number of Layers: 32 (8 x (3 Gated DeltaNet + 1 Gated Attention))
  - Intermediate Dimension: 9,216
  - LM Output: 248,320 (Padded)
  - Context Length: 262,144 natively
- **Audio input:** mono or stereo, any sample rate, up to 10 minutes per clip; several clips per question
- **Memory:** 9.9 GB in bf16; peak 13.1 GB for 10 minutes of audio

## Benchmark Results

| | Audio-Graft-4B | MOSS-Audio-4B-Instruct | Kimi-Audio-7B | Qwen2.5-Omni-7B |
|---|---|---|---|---|
| **MMAU** test-mini | 71.8 | 75.8 | 72.4 | 65.6 |
| **MMAU-Pro** | **56.9**$^{*}$ | 58.2 | 56.6 | 52.2 |

MMAU-Pro by category:

| Sound | Music | Speech | Multi-clip | Spatial | Voice chat | Open-ended | Instruction following |
|---|---|---|---|---|---|---|---|
| 46.7 | 65.7 | 53.5 | 60.2 | 46.5 | 61.7 | 65.2 | 52.9 |

Audio-Graft scored with the benchmark authors' scripts; other models: figures published by their authors (evaluation setups differ).

$^{*}$ Two-pass system with the same weights: Audio-Graft transcribes and describes the clip, then the base Qwen3.5-4B answers from that text and Audio-Graft's own answer (`audio_graft/system.py`). The single pass alone scores 52.0. The prompt style per category and the number of sampled answers (five for speech and voice chat) were chosen on MMAU-Pro.

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from audio_graft import AudioGraft

model = AudioGraft.from_pretrained("checkpoint")
print(model.ask("clip.wav", "What instruments can you hear?"))
print(model.ask(["a.wav", "b.wav"], "Which recording is louder?"))   # several clips
```

```bash
python scripts/infer.py checkpoint clip.wav "Describe the audio in two sentences."
```

For multiple-choice questions, `audio_graft.prompts.options_prompt(question, choices)` gives the layout the model was tuned on.

Download the weights (`checkpoint/`) from the latest release:

```bash
gh release download v1.0.0 --pattern 'audio-graft-4b.zip' && unzip audio-graft-4b.zip
```

## Evaluation

```bash
python scripts/eval_mmau.py     --checkpoint checkpoint --parquet test_mini.parquet
python scripts/eval_mmau_pro.py --checkpoint checkpoint --data mmau-pro --out out/single.json
python scripts/eval_mmau_pro.py --checkpoint checkpoint --data mmau-pro --out out/letters.json --prompt letters --closed_only
python scripts/eval_system.py   --checkpoint checkpoint --data mmau-pro --single out/single.json --letters out/letters.json --out out/system.json
```

The scripts write predictions for each benchmark's own evaluation script.

## Fine-tuning

```bash
python scripts/train.py --init checkpoint --data train.jsonl --out runs/mine
```

`train.jsonl` holds one `{"audio": ..., "question": ..., "answer": ...}` object per line. Only the head and the LoRA adapters are trained.

## Limitations

- Audio beyond about 10 minutes is not supported.
- Answers to multiple-choice questions are short; ask for detail explicitly.
- Weaker on emotion changes, temporal ordering and counting.

## License

Code: Apache-2.0. The base models, Qwen3-Omni AuT and Qwen3.5-4B, keep their own licenses.
