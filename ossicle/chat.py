"""Conversation about audio: Ossicle listens once, then the base Qwen3.5-4B (thinking mode) talks with you from what it heard."""
import numpy as np
import torch

FACETS = [("Genre", "What genre is this?"), ("Instruments", "What instruments are playing?"),
          ("Voice", "Is there singing or speech? If so, what is it like?"), ("Mood", "What is the mood?")]
INTRO = ("You are talking with a user about an audio recording you have listened to. Below is what you heard, in your own words (it can contain mistakes: you may have missed quiet sounds "
         "or mixed up instruments). Answer as someone who heard the recording; never mention notes, an audio model, or where your information comes from. "
         "Do not invent details you do not actually have, such as timestamps, instruments or events; if you are not sure, or you did not catch something, say so plainly. "
         "Do not just agree with a leading question: if what you heard does not support it, say that. Be conversational and answer in as much detail as the question deserves. "
         "For general questions about music, sound or anything else, use your own knowledge.\n\nWHAT YOU HEARD:\n")


def _stamp(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def listen_profile(generate, load_audio, clips, names):
    """Per-minute notes (sounds and speech) plus short answers about the whole clip. `generate(segments, question, max_new_tokens)` is the audio model."""
    lines = []
    for n, (raw, name) in enumerate(zip(clips, names), 1):
        mono = load_audio(raw, 240)[0]
        lines.append(f"Audio {n} ({name}, {len(mono) / 16000:.0f} s analysed):" if len(clips) > 1 else f"Audio ({name}, {len(mono) / 16000:.0f} s analysed):")
        for s0 in range(0, len(mono), 60 * 16000):
            seg = mono[s0: s0 + 60 * 16000]
            if len(seg) < 16000:
                continue
            sounds = generate([("audio", seg, None)], "Describe the sounds or music in this audio in two sentences.", 90).strip()
            speech = generate([("audio", seg, None)], "Transcribe any speech in this audio exactly. If there is none, say there is no speech.", 120).strip()
            lines.append(f"  [{_stamp(s0 / 16000)}-{_stamp(min(s0 + 60 * 16000, len(mono)) / 16000)}] sounds: {sounds} | speech: {speech}")
        whole = mono[: 120 * 16000]
        for label, q in FACETS:
            lines.append(f"  {label}: " + generate([("audio", whole, None)], q, 40).strip())
    return "\n".join(lines)


def chat_prompt(profile, history, message, first_impression):
    turns = []
    for k, m in enumerate(history):
        if m["role"] == "user":
            turns.append(("user", (INTRO + profile + "\n\nUSER: " if k == 0 else "") + m["content"]))
        else:
            turns.append(("assistant", m["content"]))
    last = message + (f'\n\n[Your first impression when you listened again for this question: "{first_impression}". It can be too agreeable, so check it against what you heard before relying on it. '
                      'Do not quote or mention this bracket.]' if first_impression else "")
    turns.append(("user", (INTRO + profile + "\n\nUSER: " if not history else "") + last))
    out = ""
    for role, text in turns:
        out += f"<|im_start|>{role}\n" + ("<think>\n\n</think>\n\n" if role == "assistant" else "") + text + "<|im_end|>\n"
    return out + "<|im_start|>assistant\n<think>\n"


class VLLMThinker:
    """The thinking model served by vLLM (OpenAI-compatible server), for example `vllm serve thinker-model --served-model-name thinker`."""

    def __init__(self, url="http://127.0.0.1:8000", model="thinker"):
        self.url, self.model = url.rstrip("/"), model

    def _call(self, prompt, max_tokens):
        import json, urllib.request
        req = urllib.request.Request(self.url + "/v1/completions", json.dumps({"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95}).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]

    def complete(self, prompt, thinking_tokens=250, answer_tokens=300):
        thought = self._call(prompt, thinking_tokens)
        if "</think>" in thought:
            return thought.split("</think>", 1)[1].strip()
        return self._call(prompt + thought + "\n</think>\n\n", answer_tokens).strip()


class Chat:
    """chat = model.chat("clip.wav"); chat.say("What instruments are playing?"); chat.say("Why do you think so?")"""

    def __init__(self, model, audio, reader=None, thinker=None, think_tokens=250, answer_tokens=300, first_impression=True):
        from .two_pass import Reader
        self.model, self.history, self.think, self.answer, self.impression = model, [], think_tokens, answer_tokens, first_impression
        self.audio = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        self.raw = [open(a, "rb").read() if isinstance(a, str) else a for a in self.audio]
        names = [a if isinstance(a, str) else f"audio {i + 1}" for i, a in enumerate(self.audio)]
        generate = getattr(model, "generate_text", None) or model.generate
        self.gen = lambda seg, q, n: generate(seg, q, max_new_tokens=n)
        self.profile = listen_profile(self.gen, model.load_audio, self.raw, names)
        if isinstance(thinker, str):
            reader = VLLMThinker(thinker)
        self.reader = reader or (model.thinking_model() if getattr(model, "thinking_model", None) else None) or Reader(bits=4)

    def say(self, message):
        first = ""
        if self.impression:
            first = self.model.ask(self.raw if len(self.raw) > 1 else self.raw[0], message, max_new_tokens=200).strip()
        p = chat_prompt(self.profile, self.history, message, first)
        text = self.reader.complete(p, self.think, self.answer)
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
        return text
