"""Interaural cue features for the stereo adapter (same computation as training): level and phase differences per frequency band, averaged onto the audio tokens."""
import numpy as np

SR = 16000
_BANDS = {}


def _bands(nfft=512, nb=24, lo=100, hi=7600):
    if (nfft, nb) not in _BANDS:
        f = np.fft.rfftfreq(nfft, 1 / SR)
        edges = np.geomspace(lo, hi, nb + 1)
        W = np.zeros((nb, len(f)))
        for i in range(nb):
            W[i] = ((f >= edges[i]) & (f < edges[i + 1])).astype(float)
        _BANDS[(nfft, nb)] = W / np.maximum(W.sum(1, keepdims=True), 1)
    return _BANDS[(nfft, nb)]


def cue_tokens(stereo, n_tok, hop=160, nfft=512):
    from scipy.signal import stft
    _, _, Z = stft(stereo, fs=SR, nperseg=nfft, noverlap=nfft - hop, boundary="zeros", padded=True)
    L, R = Z[0], Z[1]
    W = _bands(nfft)
    EL, ER, C = W @ (np.abs(L) ** 2), W @ (np.abs(R) ** 2), W @ (L * np.conj(R))
    ild = np.clip(10 * np.log10((EL + 1e-8) / (ER + 1e-8)), -20, 20) / 10
    ipd = np.angle(C)
    coh = np.abs(C) / np.sqrt(EL * ER + 1e-12)
    F = np.concatenate([ild, coh * np.sin(ipd), coh * (1 - np.cos(ipd))], 0).T.astype(np.float32)
    F = F * np.tile(((EL + ER) > 1e-7).T, (1, 3))
    out = np.zeros((n_tok, F.shape[1]), np.float32)
    T, k = len(F), 0
    for c in range(T // 100 + (1 if T % 100 else 0)):
        a = c * 100
        b = min(T, a + 100)
        nt = 13 if (b - a) == 100 else max(1, -(-(b - a) // 8))
        edges = np.linspace(a, b, nt + 1).round().astype(int)
        for j in range(nt):
            if k >= n_tok:
                break
            out[k] = F[edges[j]: max(edges[j] + 1, edges[j + 1])].mean(0)
            k += 1
    return out
