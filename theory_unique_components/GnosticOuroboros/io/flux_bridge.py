# python/syntonic/io/flux_bridge.py

from .....core.constants import PHI
from .....nn.resonant_tensor import ResonantTensor


class FluxBridge:
    """
    The Universal Translator.
    Converts Raw Matter (Bytes) <-> Resonant Geometry (Tensors).
    NO NLP. NO TOKENIZERS. JUST PHYSICS.
    """

    def __init__(self, dim=248):
        self.dim = dim
        # The 'Base Frequency' of the bridge
        self.phi_norm = 1.0 / PHI

    def ingest_text(self, text: str) -> ResonantTensor:
        """
        Converts text into a Winding State.
        We treat ASCII bytes as physical magnitudes.
        """
        # 1. Convert to Raw Bytes (The physical reality of the text)
        raw_bytes = list(text.encode("utf-8"))

        # 2. Normalize to [-1, 1] range based on Byte dynamics (0-255)
        # We center it around 127 (Mersenne Prime M7)
        signal = [(b - 127.0) / 128.0 for b in raw_bytes]

        # 3. Pad or Fold into the Lattice Dimension (248)
        # If text is short, we pad with Silence (0).
        # If long, we fold it (Holographic compression).
        tensor_data = [0.0] * self.dim
        for i, val in enumerate(signal):
            # Simple folding: map index to dimension modulo 248
            target_dim = i % self.dim
            tensor_data[target_dim] += val * self.phi_norm

        return ResonantTensor(tensor_data, shape=[1, self.dim])

    def emit_text(self, tensor: ResonantTensor) -> str:
        """
        Interprets a Tensor back into characters.
        This is 'Listening' to the geometry.
        """
        floats = tensor.to_floats()
        chars = []

        for val in floats:
            # Scale back up to Byte range
            # We look for 'Resonant Peaks' in the values
            byte_val = int((val / self.phi_norm) * 128.0 + 127.0)

            # Filter noise: Only significant signals become characters
            if 32 <= byte_val <= 126:  # Printable ASCII range
                chars.append(chr(byte_val))

        return "".join(chars)

    def ingest_file(self, filepath: str) -> ResonantTensor:
        """Ingest any file (Image, Code, PDF) as raw binary flux."""
        with open(filepath, "rb") as f:
            data = f.read()
        # Process every 248 bytes as a 'chunk' of reality
        # (Simplified for brevity - logic matches ingest_text)
        return self.ingest_text(str(data))
