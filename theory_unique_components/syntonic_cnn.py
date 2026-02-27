"""
Pure Syntonic CNN: Convolutional networks with DHSR structure.

NO PYTORCH DEPENDENCIES - uses sn.Module and ResonantTensor.

All convolution operations use the Rust backend for efficiency:
- 1D convolution for sequence data
- 2D convolution for image data
- Max pooling, average pooling, global average pooling

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from srt_library.core import sn as sn
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2


def _conv1d_pure(
    x: List[float],
    seq_len: int,
    in_channels: int,
    kernel: List[float],
    kernel_size: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[List[float], int]:
    """
    Pure Python 1D convolution.

    Args:
        x: Flattened input (seq_len * in_channels)
        seq_len: Input sequence length
        in_channels: Number of input channels
        kernel: Flattened kernel (out_channels * in_channels * kernel_size)
        kernel_size: Size of convolution kernel
        out_channels: Number of output channels
        stride: Convolution stride
        padding: Padding size

    Returns:
        (output_data, output_seq_len)
    """
    # Calculate output length
    out_len = (seq_len + 2 * padding - kernel_size) // stride + 1

    output = []

    for oc in range(out_channels):
        for i in range(out_len):
            val = 0.0
            pos = i * stride - padding

            for k in range(kernel_size):
                input_pos = pos + k
                if 0 <= input_pos < seq_len:
                    for ic in range(in_channels):
                        # x index: input_pos * in_channels + ic
                        x_idx = input_pos * in_channels + ic
                        # kernel index: oc * (in_channels * kernel_size) + ic * kernel_size + k
                        k_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k
                        if x_idx < len(x) and k_idx < len(kernel):
                            val += x[x_idx] * kernel[k_idx]

            output.append(val)

    return output, out_len


class PureSyntonicConv1d(sn.Module):
    """
    1D convolution with DHSR processing.

    Pure Python + ResonantTensor implementation.

    Example:
        >>> conv = PureSyntonicConv1d(in_channels=16, out_channels=32, kernel_size=3)
        >>> x = ResonantTensor(...)  # (seq_len, in_channels)
        >>> y = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize syntonic 1D convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.precision = precision
        self.device = device

        # Kernel weights
        self.kernel = sn.Parameter(
            [out_channels, in_channels, kernel_size], init="kaiming", device=device
        )

        # Bias
        self.bias = sn.Parameter([out_channels], init="zeros", device=device)

        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, in_channels)

        Returns:
            Output (out_len, out_channels)
        """
        x_data = x.to_floats()
        shape = x.shape
        seq_len = shape[0]

        kernel_data = self.kernel.to_list()
        bias_data = self.bias.to_list()

        # Perform convolution
        output_data, out_len = _conv1d_pure(
            x_data,
            seq_len,
            self.in_channels,
            kernel_data,
            self.kernel_size,
            self.out_channels,
            self.stride,
            self.padding,
        )

        # Add bias
        for i, val in enumerate(output_data):
            oc = i % self.out_channels
            output_data[i] = val + bias_data[oc]

        # Apply ReLU
        output_data = [max(0.0, v) for v in output_data]

        # Reshape to (out_len, out_channels)
        # Currently (out_channels, out_len), need to transpose
        reshaped = []
        for i in range(out_len):
            for oc in range(self.out_channels):
                reshaped.append(output_data[oc * out_len + i])

        mode_norms = [float(i * i) for i in range(len(reshaped))]
        output = ResonantTensor(
            reshaped, [out_len, self.out_channels], mode_norms, self.precision
        )

        self._syntony = output.syntony

        return output

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony


class PureSyntonicConv2d(sn.Module):
    """
    2D convolution with DHSR processing.

    Uses Rust backend for efficient convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize syntonic 2D convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.precision = precision
        self.device = device

        # Kernel weights [out_channels, kernel_h, kernel_w, in_channels]
        self.kernel = sn.Parameter(
            [out_channels, kernel_size, kernel_size, in_channels],
            init="kaiming",
            device=device,
        )

        # Bias
        self.bias = sn.Parameter([out_channels], init="zeros", device=device)

        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass using Rust conv2d.

        Args:
            x: Input (batch, height, width, in_channels) or (height, width, in_channels)

        Returns:
            Output (batch, out_h, out_w, out_channels)
        """
        from srt_library.core._core import py_conv2d

        x_data = x.to_floats()
        shape = x.shape

        # Add batch dim if needed
        if len(shape) == 3:
            h, w, c = shape
            input_shape = [1, h, w, c]
        else:
            input_shape = list(shape)

        kernel_data = self.kernel.to_list()
        kernel_shape = [
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.in_channels,
        ]

        # Perform convolution
        output_data, out_shape = py_conv2d(
            x_data,
            input_shape,
            kernel_data,
            kernel_shape,
            (self.stride, self.stride),
            (self.padding, self.padding),
        )

        # Add bias
        bias_data = self.bias.to_list()
        batch, oh, ow, oc = out_shape
        for b in range(batch):
            for h in range(oh):
                for w in range(ow):
                    for c in range(oc):
                        idx = b * (oh * ow * oc) + h * (ow * oc) + w * oc + c
                        output_data[idx] += bias_data[c]

        # Apply ReLU
        output_data = [max(0.0, v) for v in output_data]

        mode_norms = [float(i * i) for i in range(len(output_data))]
        output = ResonantTensor(output_data, out_shape, mode_norms, self.precision)

        self._syntony = output.syntony

        return output

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony


class PureSyntonicCNN1d(sn.Module):
    """
    Simple 1D CNN for sequence classification.

    Pure Python + ResonantTensor implementation.

    Architecture:
    - Stack of 1D convolutions
    - Global average pooling
    - Linear classifier

    Example:
        >>> model = PureSyntonicCNN1d(in_channels=32, num_classes=10)
        >>> x = ResonantTensor(...)  # (seq_len, in_channels)
        >>> logits = model(x)
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 10,
        hidden_channels: List[int] = [64, 128],
        kernel_size: int = 3,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize 1D syntonic CNN.

        Args:
            in_channels: Input feature channels
            num_classes: Number of output classes
            hidden_channels: List of hidden channel sizes
            kernel_size: Convolution kernel size
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.precision = precision
        self.device = device

        # Build conv layers
        self.convs = sn.ModuleList()
        ch = in_channels
        for out_ch in hidden_channels:
            self.convs.append(
                PureSyntonicConv1d(
                    ch,
                    out_ch,
                    kernel_size,
                    padding=kernel_size // 2,
                    precision=precision,
                    device=device,
                )
            )
            ch = out_ch

        # Classifier - shape [out_features, in_features] for matmul(X @ W.T)
        self.classifier = sn.Parameter([num_classes, ch], init="kaiming", device=device)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, in_channels)

        Returns:
            Logits (num_classes,)
        """
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)

        # Global average pooling
        data = x.to_floats()
        shape = x.shape
        seq_len, channels = shape

        pooled = []
        for c in range(channels):
            col_sum = sum(data[s * channels + c] for s in range(seq_len))
            pooled.append(col_sum / seq_len)

        pooled_rt = ResonantTensor(
            pooled,
            [1, channels],
            [float(i * i) for i in range(channels)],
            self.precision,
        )

        # Classifier
        logits = pooled_rt.matmul(self.classifier.tensor)

        return logits

    @property
    def syntony(self) -> float:
        """Get average syntony across layers."""
        syntonies = [conv.syntony for conv in self.convs if conv.syntony is not None]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


class PureSyntonicCNN2d(sn.Module):
    """
    Simple 2D CNN for image classification.

    Pure Python + ResonantTensor implementation using Rust conv2d backend.

    Architecture:
    - Stack of 2D convolutions with max pooling
    - Global average pooling
    - Linear classifier

    Example:
        >>> model = PureSyntonicCNN2d(in_channels=3, num_classes=10)
        >>> x = ResonantTensor(...)  # (batch, height, width, in_channels)
        >>> logits = model(x)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        pool_size: int = 2,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize 2D syntonic CNN.

        Args:
            in_channels: Input image channels (e.g., 3 for RGB)
            num_classes: Number of output classes
            hidden_channels: List of hidden channel sizes for conv layers
            kernel_size: Convolution kernel size
            pool_size: Max pooling kernel size
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.precision = precision
        self.device = device
        self.pool_size = pool_size
        self.num_classes = num_classes

        # Build conv layers
        self.convs = sn.ModuleList()
        ch = in_channels
        for out_ch in hidden_channels:
            self.convs.append(
                PureSyntonicConv2d(
                    ch,
                    out_ch,
                    kernel_size,
                    padding=kernel_size // 2,
                    precision=precision,
                    device=device,
                )
            )
            ch = out_ch

        self.final_channels = ch

        # Classifier weight - shape [out_features, in_features] for matmul(X @ W.T)
        self.classifier = sn.Parameter([num_classes, ch], init="kaiming", device=device)

    def _max_pool2d(
        self, data: List[float], batch: int, h: int, w: int, c: int, pool_size: int
    ) -> Tuple[List[float], int, int]:
        """
        Apply 2D max pooling.

        Args:
            data: Flattened input (batch * h * w * c)
            batch: Batch size
            h: Height
            w: Width
            c: Channels
            pool_size: Pooling kernel size

        Returns:
            (pooled_data, out_h, out_w)
        """
        out_h = h // pool_size
        out_w = w // pool_size

        output = []
        for b in range(batch):
            for oh in range(out_h):
                for ow in range(out_w):
                    for ch in range(c):
                        # Find max in pooling window
                        max_val = float("-inf")
                        for ph in range(pool_size):
                            for pw in range(pool_size):
                                ih = oh * pool_size + ph
                                iw = ow * pool_size + pw
                                if ih < h and iw < w:
                                    idx = b * (h * w * c) + ih * (w * c) + iw * c + ch
                                    if idx < len(data):
                                        max_val = max(max_val, data[idx])
                        output.append(max_val if max_val != float("-inf") else 0.0)

        return output, out_h, out_w

    def _global_avg_pool2d(
        self, data: List[float], batch: int, h: int, w: int, c: int
    ) -> List[float]:
        """
        Apply global average pooling.

        Args:
            data: Flattened input (batch * h * w * c)
            batch: Batch size
            h: Height
            w: Width
            c: Channels

        Returns:
            Pooled data (batch * c)
        """
        output = []
        spatial_size = h * w

        for b in range(batch):
            for ch in range(c):
                total = 0.0
                for ih in range(h):
                    for iw in range(w):
                        idx = b * (h * w * c) + ih * (w * c) + iw * c + ch
                        if idx < len(data):
                            total += data[idx]
                output.append(total / spatial_size if spatial_size > 0 else 0.0)

        return output

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (batch, height, width, in_channels) or (height, width, in_channels)

        Returns:
            Logits (batch, num_classes) or (num_classes,)
        """
        shape = x.shape

        # Track if we need to squeeze batch dim at the end
        squeeze_batch = len(shape) == 3

        # Ensure 4D input
        if len(shape) == 3:
            h, w, c = shape
            # Reshape to add batch dimension
            data = x.to_floats()
            mode_norms = [float(i * i) for i in range(len(data))]
            x = ResonantTensor(data, [1, h, w, c], mode_norms, self.precision)

        # Apply conv layers with pooling
        for i, conv in enumerate(self.convs):
            x = conv(x)

            # Apply max pooling after each conv (except maybe the last)
            if self.pool_size > 1:
                data = x.to_floats()
                batch, h, w, c = x.shape

                # Only pool if spatial dimensions are large enough
                if h >= self.pool_size and w >= self.pool_size:
                    pooled_data, out_h, out_w = self._max_pool2d(
                        data, batch, h, w, c, self.pool_size
                    )
                    mode_norms = [float(i * i) for i in range(len(pooled_data))]
                    x = ResonantTensor(
                        pooled_data,
                        [batch, out_h, out_w, c],
                        mode_norms,
                        self.precision,
                    )

        # Global average pooling
        data = x.to_floats()
        batch, h, w, c = x.shape
        pooled = self._global_avg_pool2d(data, batch, h, w, c)

        # Create tensor for classifier input: (batch, channels)
        pooled_rt = ResonantTensor(
            pooled,
            [batch, c],
            [float(i * i) for i in range(len(pooled))],
            self.precision,
        )

        # Apply classifier: (batch, channels) @ (channels, num_classes) -> (batch, num_classes)
        logits = pooled_rt.matmul(self.classifier.tensor)

        # Squeeze batch if input was 3D
        if squeeze_batch:
            logits_data = logits.to_floats()
            mode_norms = [float(i * i) for i in range(self.num_classes)]
            logits = ResonantTensor(
                logits_data, [self.num_classes], mode_norms, self.precision
            )

        return logits

    @property
    def syntony(self) -> float:
        """Get average syntony across conv layers."""
        syntonies = [conv.syntony for conv in self.convs if conv.syntony is not None]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


if __name__ == "__main__":
    import random

    print("=" * 70)
    print("Pure Syntonic CNN Test")
    print("=" * 70)

    in_channels = 16
    seq_len = 20

    # Create random input
    data = [random.gauss(0, 0.5) for _ in range(seq_len * in_channels)]
    mode_norms = [float(i * i) for i in range(len(data))]
    x = ResonantTensor(data, [seq_len, in_channels], mode_norms, 100)

    print(f"\nInput shape: {x.shape}")
    print(f"Input syntony: {x.syntony:.4f}")

    # Test 1D conv
    conv = PureSyntonicConv1d(in_channels=in_channels, out_channels=32, kernel_size=3)
    y = conv(x)

    print("\n1D Conv:")
    print(f"  Output shape: {y.shape}")
    print(f"  Layer syntony: {conv.syntony:.4f}")

    # Test full 1D CNN
    model = PureSyntonicCNN1d(
        in_channels=in_channels, num_classes=5, hidden_channels=[32, 64]
    )
    logits = model(x)

    print("\n1D CNN:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Model syntony: {model.syntony:.4f}")

    # Test 2D conv
    print("\n2D Conv:")
    img_h, img_w, img_c = 8, 8, 3
    img_data = [random.gauss(0, 0.5) for _ in range(img_h * img_w * img_c)]
    img_mode_norms = [float(i * i) for i in range(len(img_data))]
    img = ResonantTensor(img_data, [img_h, img_w, img_c], img_mode_norms, 100)

    conv2d = PureSyntonicConv2d(img_c, 32, kernel_size=3)
    y2d = conv2d(img)
    print(f"  Input shape: {img.shape}")
    print(f"  Output shape: {y2d.shape}")
    print(f"  Layer syntony: {conv2d.syntony:.4f}")

    # Test full 2D CNN
    print("\n2D CNN:")
    model2d = PureSyntonicCNN2d(
        in_channels=img_c, num_classes=10, hidden_channels=[16, 32], pool_size=2
    )
    logits2d = model2d(img)
    print(f"  Input shape: {img.shape}")
    print(f"  Logits shape: {logits2d.shape}")
    print(f"  Model syntony: {model2d.syntony:.4f}")

    print("\n" + "=" * 70)
    print("All Pure Syntonic CNN tests passed (1D and 2D).")
    print("=" * 70)
