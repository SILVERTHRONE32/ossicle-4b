"""Long audio mode: hours of audio with memory that grows only with the language model's cache.

The audio is encoded in pieces of 64 s (8 attention windows of the audio encoder, so the encoder's output is the same as for the whole clip up to
floating-point noise), each piece is projected and pushed through the language model, then freed. The attention layers' cache costs 32 KB per
audio token (12.5 tokens per second); the Gated DeltaNet layers add a fixed state of about 50 MB. Log-mel features are computed block by block
on the CPU and match the Whisper feature extractor's output for the whole clip (same STFT, filters, reflect padding and global floor of max - 8)."""
import numpy as np
import torch

PIECE_SECONDS = 64


class LongFeatures:
    def __init__(self, fe, w, piece_frames, block_frames=64000):
        self.L = len(w)
        self.T = self.L // fe.hop_length
        self.pf = piece_frames
        hop, n = fe.hop_length, fe.n_fft
        win = torch.hann_window(n)
        mel = torch.from_numpy(np.asarray(fe.mel_filters)).float()
        w = np.asarray(w, dtype=np.float32)
        self.mx, parts = -1e9, []
        for f0 in range(0, self.T, block_frames):
            f1 = min(self.T, f0 + block_frames)
            idx = np.arange(f0 * hop - n // 2, f1 * hop + n // 2)
            idx = np.where(idx < 0, -idx, idx)
            idx = np.where(idx >= self.L, 2 * (self.L - 1) - idx, idx)
            x = torch.from_numpy(w[np.clip(idx, 0, self.L - 1)])
            mag = (torch.stft(x, n, hop, window=win, center=False, return_complex=True)[..., :-1].abs() ** 2).contiguous()
            ls = torch.clamp(mel.T @ mag, min=1e-10).log10()
            self.mx = max(self.mx, float(ls.max()))
            parts.append(ls)
        self.log = torch.cat(parts, 1)

    def piece(self, k):
        f = self.log[:, k * self.pf: (k + 1) * self.pf]
        return (torch.maximum(f, torch.tensor(self.mx - 8.0)) + 4.0) / 4.0

    def __len__(self):
        return -(-self.T // self.pf)


@torch.no_grad()
def stream_embeddings(model, mono, piece_seconds=PIECE_SECONDS):
    """Yield the audio's embeddings piece by piece, (n_tokens, hidden) each."""
    pf = int(piece_seconds * 100)
    fe = getattr(model, "fe", None) or model.feature_extractor
    dtype = next(model.encoder.parameters()).dtype
    win = model.encoder.n_window_infer
    assert pf % win == 0, "pieces must be a whole number of the encoder's attention windows"
    lf = LongFeatures(fe, mono, pf)
    for k in range(len(lf)):
        f = lf.piece(k)
        h = model.encoder(input_features=f.to(model.device, dtype), feature_lens=torch.tensor([f.shape[1]], device=model.device)).last_hidden_state
        yield model.head(h.to(model.head.proj1.weight.dtype))
        del h
