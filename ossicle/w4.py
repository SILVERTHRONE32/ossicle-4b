"""Group-wise 4-bit weight-only quantisation (round-to-nearest, asymmetric, per-group scale and zero point), pure PyTorch.
Weights are stored packed (two 4-bit values per byte) and dequantised to bf16 one layer at a time inside forward().  No calibration data."""
import torch, torch.nn as nn, torch.nn.functional as F


class W4Linear(nn.Module):
    def __init__(self, lin: nn.Linear, group=128, device=None):
        super().__init__()
        w = lin.weight.data if device is None else lin.weight.data.to(device); out, inn = w.shape; g = group; assert inn % g == 0
        wg = w.float().view(out, inn // g, g); mn = wg.amin(-1, keepdim=True); mx = wg.amax(-1, keepdim=True)
        scale = ((mx - mn).clamp(min=1e-8) / 15); zero = (-mn / scale).round().clamp(0, 15)
        q = (wg / scale + zero).round().clamp(0, 15).to(torch.uint8).view(out, inn)
        self.register_buffer("packed", (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous())
        self.register_buffer("scale", scale.squeeze(-1).to(w.dtype)); self.register_buffer("zero", zero.squeeze(-1).to(w.dtype))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data, requires_grad=False)
        self.out, self.inn, self.g, self.dt = out, inn, g, w.dtype

    def dequant(self):
        p = self.packed; q = torch.stack([p & 0xF, p >> 4], -1).view(self.out, self.inn // self.g, self.g).to(self.dt)
        return ((q - self.zero.unsqueeze(-1)) * self.scale.unsqueeze(-1)).view(self.out, self.inn)

    def forward(self, x):
        return F.linear(x, self.dequant(), self.bias)


def quantize_linears(root, group=128, min_out=256, skip=(), device=None):
    """Replace nn.Linear children (recursively) by W4Linear.  Returns (n_layers, bytes_before, bytes_after)."""
    n = b0 = b1 = 0
    for name, mod in list(root.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = f"{name}.{cn}" if name else cn
            if isinstance(ch, nn.Linear) and ch.in_features % group == 0 and ch.out_features >= min_out and not any(s in full for s in skip):
                q = W4Linear(ch, group, device); b0 += ch.weight.numel() * ch.weight.element_size()
                b1 += sum(t.numel() * t.element_size() for t in q.buffers()); setattr(mod, cn, q); n += 1
    torch.cuda.empty_cache(); return n, b0, b1
