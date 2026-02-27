"""
Syntonic Pixel Operations — Pure reshape/interpolation layers.

- PixelShuffle: Sub-pixel convolution (reshape + transpose)
- Upsample: Nearest neighbor or bilinear upsampling

These are pure geometric operations with no learned parameters.

Source: CRT.md §12.2
"""

from __future__ import annotations

from typing import List, Optional

from srt_library.core import sn
from srt_library.core.nn.resonant_tensor import ResonantTensor


class PixelShuffle(sn.Module):
    """
    Rearranges elements from (B, H, W, C*r²) to (B, H*r, W*r, C).

    Pure reshape — no neural operations or theory implications.
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Args:
            x: (batch, H, W, C*r²) or (H, W, C*r²)

        Returns:
            (batch, H*r, W*r, C) or (H*r, W*r, C)
        """
        r = self.upscale_factor
        shape = x.shape
        squeeze = len(shape) == 3

        if squeeze:
            h, w, c_total = shape
            batch = 1
        else:
            batch, h, w, c_total = shape

        c = c_total // (r * r)
        assert c * r * r == c_total, f"Channels {c_total} not divisible by r²={r*r}"

        data = x.to_floats()
        out_h, out_w = h * r, w * r
        output = [0.0] * (batch * out_h * out_w * c)

        for b in range(batch):
            for ih in range(h):
                for iw in range(w):
                    for oc in range(c):
                        for rh in range(r):
                            for rw in range(r):
                                # Input channel index
                                ic = oc * r * r + rh * r + rw
                                in_idx = b * (h * w * c_total) + ih * (w * c_total) + iw * c_total + ic

                                oh = ih * r + rh
                                ow = iw * r + rw
                                out_idx = b * (out_h * out_w * c) + oh * (out_w * c) + ow * c + oc

                                if in_idx < len(data) and out_idx < len(output):
                                    output[out_idx] = data[in_idx]

        out_shape = [out_h, out_w, c] if squeeze else [batch, out_h, out_w, c]
        return ResonantTensor(output, out_shape, device=x.device)


class Upsample(sn.Module):
    """
    Spatial upsampling via nearest neighbor or bilinear interpolation.
    """

    def __init__(
        self,
        scale_factor: int = 2,
        mode: str = "nearest",
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Args:
            x: (batch, H, W, C) or (H, W, C)

        Returns:
            Upsampled tensor
        """
        shape = x.shape
        squeeze = len(shape) == 3

        if squeeze:
            h, w, c = shape
            batch = 1
        else:
            batch, h, w, c = shape

        r = self.scale_factor
        out_h, out_w = h * r, w * r
        data = x.to_floats()

        if self.mode == "nearest":
            output = self._nearest(data, batch, h, w, c, out_h, out_w)
        elif self.mode == "bilinear":
            output = self._bilinear(data, batch, h, w, c, out_h, out_w)
        else:
            raise ValueError(f"Unknown upsample mode: {self.mode}")

        out_shape = [out_h, out_w, c] if squeeze else [batch, out_h, out_w, c]
        return ResonantTensor(output, out_shape, device=x.device)

    def _nearest(
        self, data: List[float], batch: int, h: int, w: int, c: int,
        out_h: int, out_w: int,
    ) -> List[float]:
        output = []
        for b in range(batch):
            for oh in range(out_h):
                for ow in range(out_w):
                    sh = oh * h // out_h
                    sw = ow * w // out_w
                    for ch in range(c):
                        idx = b * (h * w * c) + sh * (w * c) + sw * c + ch
                        output.append(data[idx] if idx < len(data) else 0.0)
        return output

    def _bilinear(
        self, data: List[float], batch: int, h: int, w: int, c: int,
        out_h: int, out_w: int,
    ) -> List[float]:
        output = []
        for b in range(batch):
            for oh in range(out_h):
                for ow in range(out_w):
                    # Map output coords to input coords
                    src_h = (oh + 0.5) * h / out_h - 0.5
                    src_w = (ow + 0.5) * w / out_w - 0.5

                    h0 = max(0, min(int(src_h), h - 1))
                    h1 = max(0, min(h0 + 1, h - 1))
                    w0 = max(0, min(int(src_w), w - 1))
                    w1 = max(0, min(w0 + 1, w - 1))

                    fh = src_h - int(src_h)
                    fw = src_w - int(src_w)
                    fh = max(0.0, min(1.0, fh))
                    fw = max(0.0, min(1.0, fw))

                    for ch in range(c):
                        v00 = data[b * (h * w * c) + h0 * (w * c) + w0 * c + ch]
                        v01 = data[b * (h * w * c) + h0 * (w * c) + w1 * c + ch]
                        v10 = data[b * (h * w * c) + h1 * (w * c) + w0 * c + ch]
                        v11 = data[b * (h * w * c) + h1 * (w * c) + w1 * c + ch]

                        val = (v00 * (1 - fh) * (1 - fw) +
                               v01 * (1 - fh) * fw +
                               v10 * fh * (1 - fw) +
                               v11 * fh * fw)
                        output.append(val)
        return output


__all__ = [
    "PixelShuffle",
    "Upsample",
]
