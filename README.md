# Ossicle-4B

**Using the two-pass cascade (Ossicle listens, the base Qwen3.5-4B answers) is strongly recommended over the single-pass model alone: 57.0 against 52.0 on MMAU-Pro.**

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
- **Audio input:** mono or stereo, any sample rate, any length (recordings over 2 minutes are streamed through the model, see Long Audio); several clips per question
- **Memory:** 9.9 GB in bf16, 3.6 GB with 4-bit weights; long recordings add about 2.5 GB per hour of audio

## Benchmark Results

| | Ossicle-4B | MOSS-Audio-4B-Instruct | Kimi-Audio-7B | Qwen2.5-Omni-7B |
|---|---|---|---|---|
| **MMAU** test-mini | 71.8 | 75.8 | 72.4 | 65.6 |
| **MMAU-Pro** | **57.0**$^{\dagger}$ | 54.9* | 56.6 | 52.2 |

MMAU-Pro by category:

| Sound | Music | Speech | Multi-clip | Spatial | Voice chat | Open-ended | Instruction following |
|---|---|---|---|---|---|---|---|
| 46.7 | 65.7 | 53.5 | 60.2 | 46.5 | 61.7 | 65.2 | 52.9 |

Ossicle scored with the benchmark authors' scripts; other models: figures published by their authors (evaluation setups differ).

\* Reproduced locally: MOSS-Audio-4B-Instruct run with its own prompting and scored with the same MMAU-Pro script.

$^{\dagger}$ Two-pass system with the same weights: Ossicle transcribes and describes the clip, then the base Qwen3.5-4B answers from that text and Ossicle's own answer (`ossicle/system.py`). The exact score is 56.97; the single pass alone scores 52.0. The prompt style per category and the number of sampled answers (five for speech and voice chat) were chosen on MMAU-Pro.

## Long Audio

Long audio mode is part of this release. Recordings longer than 2 minutes are streamed through the model in 64-second pieces: each piece is encoded, passed through the language model and freed, so memory grows only with the language model's attention cache (about 2.5 GB per hour of audio). There is nothing to switch on: `model.ask("long.wav", question)` takes this path by itself. The log-mel features are identical to the whole-clip path and the encoder output matches it up to bf16 rounding; clips of 2 minutes or less take the original path.

Longest audio (minutes) that ran to an answer, with peak GPU memory in GB in brackets, measured on the cards themselves (no memory cap). Found by doubling and then bisecting: the next 5 minutes ran out of memory. A real 10-minute clip tiled to length, greedy decoding, 60 new tokens. INT4 is `model.to_4bit()`: 4-bit weights for the language model and the audio encoder, round to nearest, groups of 128.

| | Long audio mode | Whole clip at once |
|---|---|---|
| RTX 5060 Ti, 16 GB, bf16 | **85** (13.5) | 30 (14.1) |
| RTX 5060 Ti, 16 GB, INT4 | **210** (12.4) | 80 (13.9) |
| RX 7900 XT, 20 GB, bf16 | **180** (17.5) | 65 (18.8) |
| RX 7900 XT, 20 GB, INT4 | **295** (15.9) | 95\* (15.8) |

\* On the AMD card a single pass over more than about 87 minutes also hits a ROCm convolution bug that returns wrong values without an error, so this is a memory figure only. Long audio mode never reaches it.

For comparison, the same measurement on the 20 GB card for other models, run earlier with their authors' code on the whole clip (4-bit with the same scheme), in coarser steps of 5 to 60 minutes, so these are lower bounds:

| | bf16 | 4-bit |
|---|---|---|
| **Ossicle-4B**, whole clip at once | 65 | 95\* |
| MOSS-Audio-4B | 22 | 36 |
| Qwen2.5-Omni-3B | 20 | 20 |
| MOSS-Audio-8B | 5 | 20 |
| Qwen2.5-Omni-7B | 1 | 10 |

**Many clips in one question.** The same cards and the same long audio mode, with the question holding N clips of 30 seconds each (a typical benchmark clip, labelled "Audio 1:", "Audio 2:", and so on). Long audio mode starts once the clips add up to more than 2 minutes in total. N was searched until the card ran out of memory; the next 5 clips did not fit:

| | bf16 | INT4 |
|---|---|---|
| RTX 5060 Ti, 16 GB | **190** clips (95 min) | **370** clips (185 min) |
| RX 7900 XT, 20 GB | **365** clips (182 min) | **575** clips (288 min) |

The totals differ a little from the single-recording table because the clips are fed in 30-second steps instead of 64-second pieces, which changes the activation peak.

Reproduce a row with `python scripts/bench_long.py checkpoint clip.wav --quant int4 --mode long` (add `--clip_seconds 30` to count clips instead).

This measures memory, not comprehension. The released checkpoint was trained on clips of at most 30 seconds, and in our earlier tests it lost spoken detail in recordings longer than about 10 minutes, so use long audio mode for what happens in a recording (sounds, music, a change of speaker or mood), not to transcribe hours of speech. Stereo cues are not used for long clips, and chat mode still listens to the first 4 minutes of each recording.

## Quickstart

```bash
pip install -r requirements.txt
```

**Talk to it.** Chat mode listens to the clip once, then holds a conversation about it (the thinking model is loaded in 4-bit, about 6 GB extra):

```python
from ossicle import Ossicle

model = Ossicle.from_pretrained("checkpoint")
chat = model.chat("clip.wav")                     # a path, or a list of paths
print(chat.say("What instruments can you hear?"))
print(chat.say("Why do you think so?"))               # remembers the conversation
```

**Fast mode.** One short answer straight from the audio model, no memory, about 3 seconds:

```python
print(model.ask("clip.wav", "What genre is this?"))
print(model.ask(["a.wav", "b.wav"], "Which recording is louder?"))   # several clips
```

**Long audio.** Recordings of any length work the same way; clips over 2 minutes are streamed automatically. Optional 4-bit weights (9.9 GB to 3.6 GB) fit longer recordings:

```python
print(model.ask("podcast.mp3", "What is this episode about?"))
model.to_4bit()      # call once, before asking; merges the adapters, so not for training
```

**Two-pass system.** The setting behind the MMAU-Pro score is in `ossicle/system.py` and `scripts/eval_system.py`: the audio model transcribes and describes, then the base Qwen3.5-4B answers. The Hugging Face checkpoint exposes it as `model.ask_two_pass(...)`.

For multiple-choice questions, `ossicle.prompts.options_prompt(question, choices)` gives the layout the model was tuned on.

Download the weights (`checkpoint/`) from the latest release:

```bash
gh release download v1.0.0 --pattern 'ossicle-4b.zip' && unzip ossicle-4b.zip
```

## Quantized: NVFP4 and INT4 in vLLM

Ossicle-4B ships in two quantized versions, each one checkpoint that runs entirely in vLLM: audio encoder, head, stereo adapter and language model, with the audio encoder running inside vLLM through a small plugin (`nvfp4/vllm_plugin`). Each folder also holds a quantized Qwen3.5-4B (`thinker/`) for the two-model chat, the plugin, the chat template and the chat client.

- [`nvfp4/`](https://huggingface.co/SILVERTHRONE/Ossicle-4B/tree/main/nvfp4) in the [Ossicle-4B](https://huggingface.co/SILVERTHRONE/Ossicle-4B) repository: NVFP4 (weights and activations) with NVIDIA ModelOpt, 4.6 GB. Needs a Blackwell GPU for the native FP4 kernels.
- [`int4/`](https://huggingface.co/SILVERTHRONE/Ossicle-4B/tree/main/int4) in the same repository: INT4 weights (groups of 128) with GPTQ, compressed-tensors format, 4.4 GB.

MMAU test-mini, the benchmark's own evaluation script, and the 325 single-clip spatial questions of MMAU-Pro (stereo input, options matched by text, our own scoring):

| Version | Sound | Music | Speech | Overall | Change | Spatial |
|---|---|---|---|---|---|---|
| bf16 | 73.9 | 74.3 | 67.3 | 71.8 | | |
| NVFP4, vLLM | 73.3 | 72.8 | 65.8 | 70.6 | -1.2 | 33.2 |
| INT4 GPTQ, vLLM | 74.8 | 74.9 | 66.1 | 71.9 | +0.1 | 33.5 |
| INT4 round to nearest, vLLM | 71.8 | 72.5 | 66.1 | 70.1 | -1.7 | 34.2 |
| NVFP4 without the stereo adapter, vLLM | 72.7 | 71.9 | 65.5 | 70.0 | -1.8 | 24.0 |
| NVFP4, weights only, PyTorch | | | | 71.5 | -0.3 | |

A 1,000-question benchmark has about 1.4 points of noise, so INT4 GPTQ is a no-loss result and NVFP4 is within a point of bf16. On one RTX 5060 Ti the NVFP4 checkpoint answers all 1,000 test-mini questions in 28 seconds and the INT4 checkpoint in 59 seconds (2,348 seconds for the NVFP4 weights unpacked in PyTorch). The stereo adapter works inside vLLM: spatial accuracy rises from 24.0 to about 33 to 34, the same gain as in PyTorch. Stereo goes through the offline API as a `(channels, samples)` array; `vllm serve` decodes audio URLs to mono by default and stereo through the server is not tested.

The two-pass system's thinking model, Qwen3.5-4B, was quantized the same ways and compared with its bf16 original and a community NVFP4 build (AxionML) on MMAU-Pro speech (891 questions) and voice chat (290), token-F1 scoring of the final answers, one sample each:

| Thinking model | Speech | Voice chat |
|---|---|---|
| bf16 | 51.6 | 58.6 |
| NVFP4, ModelOpt (ours) | 52.7 | 57.9 |
| NVFP4, AxionML community build | 52.2 | 56.2 |
| INT4 GPTQ | 51.6 | 59.0 |
| INT4 round to nearest | 52.1 | 59.7 |

All of them are within the noise (about 1.5 points) of the bf16 model.

Only vLLM is tested. The plugin that runs the audio encoder is written for vLLM; the language model and thinker are standard ModelOpt NVFP4 and compressed-tensors W4A16 checkpoints, which other engines may load, but we have not tried any.

Build your own from a checkpoint (NVFP4 needs `pip install nvidia-modelopt[hf]`, INT4 needs `pip install llmcompressor`):

```bash
python nvfp4/merge_lm.py    --checkpoint checkpoint --out lm_bf16
python nvfp4/calibration.py --checkpoint checkpoint --data calib.jsonl --out calibration.pt

python nvfp4/quantize.py    --lm lm_bf16 --calibration calibration.pt --out lm_nvfp4
python nvfp4/assemble.py    --checkpoint checkpoint --lm lm_nvfp4 --out Ossicle-4B/nvfp4
python nvfp4/quantize_thinker.py --calib calib_text.json --out Ossicle-4B/nvfp4/thinker

python int4/quantize_gptq.py --lm lm_bf16 --calibration calibration.pt --out lm_int4        # or int4/quantize.py for round to nearest
python nvfp4/assemble.py     --checkpoint checkpoint --lm lm_int4 --out Ossicle-4B/int4
python int4/quantize_thinker.py --calib calib_text.json --out Ossicle-4B/int4/thinker
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

- Long audio mode keeps memory flat, not accuracy: spoken detail in recordings longer than about 10 minutes is unreliable (see Long Audio).
- Fast mode gives short answers, has no memory of earlier questions, and can agree with leading questions; use chat mode for conversation.
- Chat mode answers from what the audio model heard, so fine acoustic detail is only as good as that, and a reply takes 30 to 60 seconds with the 4-bit thinking model.
- Weaker on emotion changes, temporal ordering and counting.

## License

Code: Apache-2.0. The base models, Qwen3-Omni AuT and Qwen3.5-4B, keep their own licenses.
