"""Stereo (spatial) input: interaural cues feed a small zero-initialised adapter whose output is added to the audio embeddings.
For identical channels the adapter output is exactly zero, so mono behaviour is unchanged."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
_B = {}


class StereoAdapter(nn.Module):
    def __init__(self, in_dim=72, hidden=512, out_dim=2560):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, f):
        zero = torch.zeros_like(f[:1])
        return self.fc2(F.gelu(self.fc1(f))) - self.fc2(F.gelu(self.fc1(zero)))


def bands(nfft=512, nb=24, lo=100, hi=7600):
    key = (nfft, nb)
    if key not in _B:
        f = np.fft.rfftfreq(nfft, 1 / SR); edges = np.geomspace(lo, hi, nb + 1); W = np.zeros((nb, len(f)))
        for i in range(nb): W[i] = ((f >= edges[i]) & (f < edges[i + 1])).astype(float)
        _B[key] = W / np.maximum(W.sum(1, keepdims=True), 1)
    return _B[key]

def cue_features(stereo, hop=160, nfft=512):
    """stereo (2, T) float -> (frames, 72): ILD (dB/10, clipped), w*sin(IPD), w*(1-cos(IPD)) per band.  Exactly zero when the two channels are identical."""
    from scipy.signal import stft
    _, _, Z = stft(stereo, fs=SR, nperseg=nfft, noverlap=nfft - hop, boundary="zeros", padded=True); L, R = Z[0], Z[1]; W = bands(nfft)
    EL = W @ (np.abs(L) ** 2); ER = W @ (np.abs(R) ** 2); C = W @ (L * np.conj(R))
    ild = np.clip(10 * np.log10((EL + 1e-8) / (ER + 1e-8)), -20, 20) / 10; ipd = np.angle(C); coh = np.abs(C) / np.sqrt(EL * ER + 1e-12)
    F = np.concatenate([ild, coh * np.sin(ipd), coh * (1 - np.cos(ipd))], 0).T.astype(np.float32); act = ((EL + ER) > 1e-7).T                   # silent bands carry no cue
    F = F * np.tile(act, (1, 3)); return F

def pool_tokens(F, n_tok):
    """Average the 100 fps cue frames into the audio tokens of the tower (13 tokens per full 100-frame chunk, last chunk proportionally)."""
    out = np.zeros((n_tok, F.shape[1]), np.float32); T = len(F); full = T // 100; k = 0
    for c in range(full + (1 if T % 100 else 0)):
        a = c * 100; b = min(T, a + 100); nt = 13 if (b - a) == 100 else max(1, -(-(b - a) // 8))
        edges = np.linspace(a, b, nt + 1).round().astype(int)
        for j in range(nt):
            if k >= n_tok: break
            seg = F[edges[j]: max(edges[j] + 1, edges[j + 1])]; out[k] = seg.mean(0); k += 1
    return out


def cue_tokens(stereo, n_tokens):
    """(2, T) float32 at 16 kHz -> (n_tokens, 72) interaural cue features aligned with the audio tokens."""
    return pool_tokens(cue_features(stereo), n_tokens)
