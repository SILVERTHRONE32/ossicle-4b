"""Audio-Graft: a frozen Qwen3-Omni audio encoder, a small trained head, and Qwen3.5-4B with LoRA adapters."""
import io
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER = os.environ.get("TOWER_PATH", "Atotti/Qwen3-Omni-AudioTransformer")
LLM = os.environ.get("LLM_PATH", "Qwen/Qwen3.5-4B")

PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EOS = "<|im_end|>"


class AudioHead(nn.Module):
    """Projects encoder features (1280-d) to the language model's embedding size."""

    def __init__(self, enc_proj1: nn.Linear, out_dim: int, init_std: float = 0.02):
        super().__init__()
        d = enc_proj1.in_features
        self.proj1 = nn.Linear(d, d)
        self.proj1.load_state_dict(enc_proj1.state_dict())
        self.proj2 = nn.Linear(d, out_dim)
        nn.init.normal_(self.proj2.weight, std=init_std / d ** 0.5)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        return self.proj2(F.gelu(self.proj1(x)))


def load_audio(src, max_seconds=600):
    """File path or bytes -> (mono 16 kHz float32, stereo (2, T) or None if the channels are identical)."""
    data = open(src, "rb").read() if isinstance(src, (str, os.PathLike)) else src
    w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    w = w.T[:, : int(max_seconds * sr)]
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(16000, sr)
        w = resample_poly(w, 16000 // g, sr // g, axis=-1).astype(np.float32)
    mono = w.mean(0)
    stereo = None
    if w.shape[0] >= 2 and np.sqrt(np.mean((w[0] - w[1]) ** 2)) > 1e-3 * (np.sqrt(np.mean(mono ** 2)) + 1e-8):
        stereo = w[:2].astype(np.float32)
    return mono.astype(np.float32), stereo


class AudioGraft(nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder, _get_feat_extract_output_lengths)

        self._out_len = _get_feat_extract_output_lengths
        self.fe = WhisperFeatureExtractor.from_pretrained(ENCODER)
        self.tok = AutoTokenizer.from_pretrained(LLM)
        enc = Qwen3OmniMoeAudioEncoder.from_pretrained(ENCODER, dtype=dtype)
        proj1 = enc.proj1
        enc.proj1, enc.act, enc.proj2 = nn.Identity(), nn.Identity(), nn.Identity()
        self.encoder = enc.eval().requires_grad_(False)
        full = Qwen3_5ForConditionalGeneration.from_pretrained(LLM, dtype=dtype)
        self.llm = full.model.language_model
        self.lm_head = full.lm_head
        del full.model.visual
        self.llm.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        std = self.llm.embed_tokens.weight.float().std().item()
        self.head = AudioHead(proj1, self.llm.config.hidden_size, std).float()
        self.stereo = None
        self.to(device)
        self.device, self.dtype = device, dtype

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path, device="cuda", dtype=torch.bfloat16, trainable=False):
        """`path` holds head.pt, lora/ and (optionally) stereo_adapter.pt."""
        from peft import PeftModel
        from .stereo import StereoAdapter
        m = cls(device, dtype)
        m.head.load_state_dict(torch.load(f"{path}/head.pt", map_location=device))
        m.llm = PeftModel.from_pretrained(m.llm, f"{path}/lora", is_trainable=trainable)
        m.llm.train(trainable) if trainable else m.llm.eval()
        if os.path.exists(f"{path}/stereo_adapter.pt"):
            m.stereo = StereoAdapter(out_dim=m.llm.config.hidden_size).to(device).float()
            m.stereo.load_state_dict(torch.load(f"{path}/stereo_adapter.pt", map_location=device))
            m.stereo.eval()
        return m

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.head.state_dict(), f"{path}/head.pt")
        self.llm.save_pretrained(f"{path}/lora")
        if self.stereo is not None:
            torch.save(self.stereo.state_dict(), f"{path}/stereo_adapter.pt")

    # ---- audio ---------------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, mono):
        out = self.fe([mono], sampling_rate=16000, return_attention_mask=True, padding="longest",
                      truncation=False, return_tensors="pt")
        n = out["attention_mask"].sum(-1)
        feats = out["input_features"][0, :, : n[0]]
        return self.encoder(input_features=feats.to(self.device, self.dtype),
                            feature_lens=n.to(self.device)).last_hidden_state

    def embed(self, mono, stereo=None):
        """One clip -> (n_tokens, hidden) embeddings at 12.5 tokens per second."""
        e = self.head(self._encode(mono).to(self.head.proj1.weight.dtype))
        if stereo is not None and self.stereo is not None:
            from .stereo import cue_tokens
            e = e + self.stereo(torch.tensor(cue_tokens(stereo, e.shape[0]), device=e.device, dtype=torch.float32)).to(e.dtype)
        return e

    # ---- sequences -----------------------------------------------------------------------
    def _ids(self, s):
        return torch.tensor(self.tok(s, add_special_tokens=False)["input_ids"], device=self.device)

    def build(self, examples):
        """examples: dicts with 'segments' [('text', str) | ('audio', mono, stereo) | ('embedding', tensor)], 'question', optional 'answer'.
        Returns inputs_embeds, attention_mask, answer_mask, target_ids."""
        emb = self.llm.embed_tokens
        pre = emb(self._ids(PREFIX))
        seqs, spans = [], []
        for ex in examples:
            parts = [pre]
            for s in ex["segments"]:
                parts.append(emb(self._ids(s[1])) if s[0] == "text" else s[1].to(pre.dtype) if s[0] == "embedding" else self.embed(s[1], s[2]).to(pre.dtype))
            parts.append(emb(self._ids(ex["question"] + SUFFIX)))
            t = self._ids(ex["answer"] + EOS) if ex.get("answer") is not None else None
            if t is not None:
                parts.append(emb(t))
            seqs.append(torch.cat(parts, 0))
            spans.append(t)
        L, H = max(s.shape[0] for s in seqs), seqs[0].shape[1]
        x = pre.new_zeros(len(seqs), L, H)
        attn = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        mask = torch.zeros(len(seqs), L, dtype=torch.bool, device=self.device)
        tgt = torch.full((len(seqs), L), -100, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            x[i, : s.shape[0]] = s
            attn[i, : s.shape[0]] = 1
            if spans[i] is not None:
                n = len(spans[i])
                mask[i, s.shape[0] - n: s.shape[0]] = True
                tgt[i, s.shape[0] - n: s.shape[0]] = spans[i]
        return x, attn, mask, tgt

    def loss(self, examples):
        x, attn, mask, tgt = self.build(examples)
        h = self.llm(inputs_embeds=x, attention_mask=attn, use_cache=False).last_hidden_state
        logits = self.lm_head(h[:, :-1][mask[:, 1:]]).float()
        return F.cross_entropy(logits, tgt[:, 1:][mask[:, 1:]])

    @torch.no_grad()
    def generate(self, segments, question, max_new_tokens=200):
        """Greedy answer for one example."""
        x, _, _, _ = self.build([{"segments": segments, "question": question}])
        o = self.llm(inputs_embeds=x, use_cache=True)
        past, nxt = o.past_key_values, self.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
        eos, ids = self.tok.convert_tokens_to_ids(EOS), []
        for _ in range(max_new_tokens):
            if nxt.item() == eos:
                break
            ids.append(nxt.item())
            o = self.llm(inputs_embeds=self.llm.embed_tokens(nxt)[:, None], past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, self.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
        return self.tok.decode(ids)

    def ask(self, audio, question, max_new_tokens=200):
        """`audio`: a path, bytes, or a list of them (several clips are labelled 'Audio 1:', 'Audio 2:', ...)."""
        clips = [load_audio(a) for a in (audio if isinstance(audio, (list, tuple)) else [audio])]
        if len(clips) == 1:
            segs = [("audio", *clips[0])]
        else:
            segs = []
            for i, c in enumerate(clips):
                segs += [("text", ("\n" if i else "") + f"Audio {i + 1}:\n"), ("audio", *c)]
            segs.append(("text", "\n"))
        return self.generate(segs, question, max_new_tokens)
