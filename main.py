import argparse
import os
import numpy as np
from PIL import Image
from compression import compress
from decompression import decompress

p = argparse.ArgumentParser(description="Vector Quantization image compressor")
sub = p.add_subparsers(dest="mode", required=True)

c = sub.add_parser("compress")
c.add_argument("image")
c.add_argument("--h", type=int, default=4, help="block height")
c.add_argument("--w", type=int, default=4, help="block width")
c.add_argument("--k", type=int, default=256, help="codebook size")
c.add_argument("--cb", default="codebook.bin")
c.add_argument("--out", default="compressed.bin")

d = sub.add_parser("decompress")
d.add_argument("--cb", default="codebook.bin")
d.add_argument("--inp", default="compressed.bin")
d.add_argument("--out", default="reconstructed.png")
d.add_argument("--orig", help="original image, to print the MSE")

a = p.parse_args()

if a.mode == "compress":
    compress(a.image, a.h, a.w, a.k, a.cb, a.out)
    orig = np.array(Image.open(a.image).convert('L')).size   # 8 bits/pixel -> bytes
    comp = os.path.getsize(a.cb) + os.path.getsize(a.out)
    print(f"Done. Original {orig} bytes, compressed {comp} bytes, ratio {orig / comp:.2f}:1")
else:
    decompress(a.cb, a.inp, a.out)
    print("Saved", a.out)
    if a.orig:
        o = np.array(Image.open(a.orig).convert('L'), dtype=float)
        r = np.array(Image.open(a.out), dtype=float)
        print("MSE:", np.mean((o - r) ** 2))