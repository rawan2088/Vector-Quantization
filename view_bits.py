import struct
import sys
import numpy as np

with open(sys.argv[1], "rb") as f:
    h, w, K = struct.unpack('<BBH', f.read(4))
    codebook = np.frombuffer(f.read(), dtype=np.uint8)

codebook = codebook.reshape(K, h * w)

np.set_printoptions(threshold=np.inf, linewidth=150)

print(f"Block dimensions: {h}x{w}")
print(f"Number of vectors: {K}")
print(f"Vector dimensions: {h*w}")
print("\nCodebook:")
print(codebook)