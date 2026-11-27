import os
import numpy as np


# ---------- Quality ----------
def mse(original, reconstructed):
    # Mean Squared Error: average of the squared pixel differences.
    # Convert to float first: uint8 subtraction wraps around (3 - 5 = 254).
    diff = original.astype(float) - reconstructed.astype(float)
    return float(np.mean(diff ** 2))


def theoretical_ratio(H, W, h, w, K):
    # Lecture formula (no headers, no byte rounding). All sizes are in bits.
    blocks = -(-H // h) * -(-W // w)          # ceiling division counts padded blocks
    label_bits = max(1, (K - 1).bit_length())  # same as log2(K) for a power of 2
    original = H * W * 8
    labels = blocks * label_bits
    codebook = K * h * w * 8
    return original, labels, codebook, original / (labels + codebook)