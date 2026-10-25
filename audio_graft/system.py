"""Two-pass system: Audio-Graft transcribes and describes the clip, then the base Qwen3.5-4B (adapters off, thinking mode) answers from the text."""
import torch

from .model import LLM

TRANSCRIBE = "Transcribe the speech in this audio exactly."
DESCRIBE = "Describe, in two or three sentences, the sounds or music in this audio other than the speech."


def listen(model, mono):
    """Pass 1: (transcript, description) of one clip."""
    w = [("audio", mono, None)]
    return (model.generate(w, TRANSCRIBE, max_new_tokens=300), model.generate(w, DESCRIBE, max_new_tokens=100))


def prompt(kind, transcript, description, question=None, choices=None, draft=None):
    """Pass-2 prompt. kind: 'spoken_question' (answer a spoken multiple-choice question), 'spoken_instruction'
    (follow a spoken instruction), or 'with_draft' (question about the audio, with Audio-Graft's own answer as a third source)."""
    t, d = transcript.strip(), description.strip()
    if kind == "spoken_question":
        opts = "\n".join(f"- {c}" for c in choices)
        u = (f'A question was spoken aloud. This is an automatic transcript of it (it may contain small errors):\n"{t}"\n\n'
             f"Choose the option that correctly answers the spoken question.\nOptions:\n{opts}\n\n"
             "Think it through, then reply with the exact text of the one correct option and nothing else.")
    elif kind == "spoken_instruction":
        u = (f'A spoken instruction was transcribed automatically (it may contain small errors):\n"{t}"\n\n'
             f"Automatic description of the other audio content, if any: {d}\n\n"
             "Follow the spoken instruction exactly, including every formatting constraint it states, and write your response now.")
    else:
        ctx = (f'You cannot hear an audio clip, but you have three sources about it.\n(1) Automatic transcript of its speech (may contain errors): "{t}"\n'
               f"(2) Automatic description of its other sounds: {d}\n"
               f'(3) The answer an audio model gave directly from the sound; it hears tone, voices and sounds that a transcript loses, and it is often but not always right: "{str(draft).strip()}"\n\n')
        if choices is None:
            u = ctx + f"Question: {question}\n\nWeigh the three sources, think it through, then give a helpful, accurate answer in a few sentences."
        else:
            opts = "\n".join(f"- {c}" for c in choices)
            u = (ctx + f"Question: {question}\nOptions:\n{opts}\n\n"
                 "Weigh the three sources, think it through, then reply with the exact text of the one correct option and nothing else.")
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class Reader:
    """Pass 2: the base text model, sampled (T 0.6) with a thinking budget and a forced close."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(LLM)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(LLM, dtype=dtype, device_map={"": device}).eval()
        self.device = device

    def _gen(self, prompts, max_new):
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_p=0.95,
                                    pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def answer(self, prompts, thinking_budget=900, answer_budget=120):
        outs = self._gen(prompts, thinking_budget)
        redo = [i for i, o in enumerate(outs) if "</think>" not in o]
        closed = self._gen([prompts[i] + outs[i] + "\n</think>\n\n" for i in redo], answer_budget) if redo else []
        fin = dict(zip(redo, closed))
        return [(o.split("</think>")[-1] if "</think>" in o else fin[i]).strip() for i, o in enumerate(outs)]
