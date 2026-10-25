"""NVFP4 weight quantization: FP4 (E2M1) values, one FP8 (E4M3) scale per 16 weights, one FP32 per-tensor scale.
Tensor names follow the NVIDIA ModelOpt checkpoint layout (weight, weight_scale, weight_scale_2).
Weights are stored packed (two 4-bit codes per byte, even index in the low nibble) and dequantised layer by layer inside forward().
Round-to-nearest, no calibration. The per-tensor scale is a power of two so it is exact in any dtype."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK = 16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
LUT = torch.cat([E2M1, -E2M1])                                   # code = sign bit (8) | magnitude index
MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])    # round-to-nearest boundaries between E2M1 magnitudes


def quantize_tensor(w):
    """w: (out, in) float tensor, in divisible by 16 -> packed uint8 (out, in/2), scales float8_e4m3fn (out, in/16), gscale float32 scalar."""
    out, inn = w.shape
    wf = w.float().view(out, inn // BLOCK, BLOCK)
    gmax = wf.abs().max().clamp(min=1e-30)
    gscale = 2.0 ** math.ceil(math.log2(float(gmax) / (6.0 * 448.0)))        # smallest power of two that keeps block scales inside E4M3
    bscale = ((wf.abs().amax(-1) / 6.0) / gscale).clamp(max=448.0).to(torch.float8_e4m3fn)
    s = bscale.float() * gscale                                              # the scale actually used (after E4M3 rounding)
    x = wf / s.unsqueeze(-1).clamp(min=1e-30)
    mid = MIDS.to(x.device)
    idx = torch.bucketize(x.abs().clamp(max=6.0), mid)                       # 0..7
    code = (idx | ((x < 0) & (idx > 0)).to(idx.dtype) * 8).to(torch.uint8).view(out, inn)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, bscale, torch.tensor(gscale, dtype=torch.float32)


_TABLES = {}


def _byte_table(device):
    """256 x 2 table: the two E2M1 values stored in a byte (low nibble first), so one gather decodes two weights."""
    if device not in _TABLES:
        b = torch.arange(256)
        _TABLES[device] = torch.stack([LUT[b & 15], LUT[b >> 4]], -1).to(device)
    return _TABLES[device]


def dequantize(packed, bscale, gscale, dtype=torch.bfloat16):
    out = packed.shape[0]
    vals = _byte_table(packed.device)[packed.long()].view(out, -1)
    s = bscale.float() * gscale.float()
    return (vals.view(out, -1, BLOCK) * s.unsqueeze(-1)).view(out, -1).to(dtype)


class NVFP4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.in_features, self.out_features, self.dt = in_features, out_features, dtype
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, in_features // BLOCK, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.ones((), dtype=torch.float32, device=device), persistent=False)   # only used by W4A4 runtimes
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_linear(cls, lin, device=None):
        w = lin.weight.data if device is None else lin.weight.data.to(device)
        q = cls(lin.in_features, lin.out_features, lin.bias is not None, w.dtype, device=w.device)
        q.weight, q.weight_scale, q.weight_scale_2 = [t.to(w.device) for t in quantize_tensor(w)]
        if lin.bias is not None:
            q.bias = nn.Parameter(lin.bias.data.to(w.device), requires_grad=False)
        return q

    def forward(self, x):
        return F.linear(x, dequantize(self.weight, self.weight_scale, self.weight_scale_2, x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def quantize_model(root, skip=(), min_out=256, device=None, only=None):
    """Replace nn.Linear children by NVFP4Linear. `skip`: substrings of module names to keep in 16-bit (embeddings, lm_head).
    `only`: if given, only modules whose name contains one of these substrings. Returns the number of layers converted."""
    n = 0
    for name, mod in list(root.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = f"{name}.{cn}" if name else cn
            if isinstance(ch, nn.Linear) and ch.in_features % BLOCK == 0 and ch.out_features >= min_out and not any(s in full for s in skip) \
                    and (only is None or any(o in full for o in only)):
                setattr(mod, cn, NVFP4Linear.from_linear(ch, device)); n += 1
    return n


def as_quantized_shell(root, skip=(), min_out=256, only=None):
    """Same replacement on a (meta-device) model with no data, so a saved NVFP4 state dict can be loaded into it."""
    n = 0
    for name, mod in list(root.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = f"{name}.{cn}" if name else cn
            if isinstance(ch, nn.Linear) and ch.in_features % BLOCK == 0 and ch.out_features >= min_out and not any(s in full for s in skip) \
                    and (only is None or any(o in full for o in only)):
                setattr(mod, cn, NVFP4Linear(ch.in_features, ch.out_features, ch.bias is not None, ch.weight.dtype, device=ch.weight.device)); n += 1
    return n
