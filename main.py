import argparse
import os
import struct
import sys
import time
import numpy as np
from PIL import Image
from compression import compress, next_pow2
from decompression import decompress
from metrics import mse, theoretical_ratio

MAX_K = 32768  # K is stored in 2 bytes, so the largest usable power of 2 is 2^15
LOSSLESS = ('.png', '.bmp', '.pgm', '.tif', '.tiff')


# ---------- File checks ----------
def load_gray(path):
    with Image.open(path) as im:
        return np.array(im.convert('L'))


def require_file(path):
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")


def check_outputs(outputs, inputs):
    # Never overwrite an input, and never write two outputs to the same file.
    real = [os.path.realpath(p) for p in outputs]
    if len(set(real)) != len(real):
        raise ValueError("The output files must have different names.")
    if set(real) & {os.path.realpath(p) for p in inputs}:
        raise ValueError("An output file would overwrite an input file.")
    for p in real:
        if not os.path.isdir(os.path.dirname(p)):
            raise ValueError(f"Output folder does not exist: {os.path.dirname(p)}")


def read_header(cb_file, bin_file):
    # Verify that both files are complete and belong to a compatible pair.
    with open(cb_file, 'rb') as f:
        head = f.read(4)
    if len(head) < 4:
        raise ValueError("Codebook file is too short.")
    h, w, K = struct.unpack('<BBH', head)
    if min(h, w, K) < 1 or os.path.getsize(cb_file) != 4 + K * h * w:
        raise ValueError("Invalid or damaged codebook file (check the file order).")

    with open(bin_file, 'rb') as f:
        head = f.read(8)
        data = f.read()
    if len(head) < 8:
        raise ValueError("Compressed file is too short.")
    H, W = struct.unpack('<II', head)
    if min(H, W) < 1:
        raise ValueError("Invalid compressed file.")
    count = -(-H // h) * -(-W // w)
    bits = max(1, (K - 1).bit_length())
    if len(data) != (count * bits + 7) // 8:
        raise ValueError("Compressed file is damaged or does not match this codebook.")
    return H, W, h, w, K


# ---------- The two operations ----------
def run_compress(image, h, w, K, cb_file, bin_file):
    require_file(image)
    if not (1 <= h <= 255 and 1 <= w <= 255):
        raise ValueError("Block height and width must be between 1 and 255.")
    if not 1 <= K <= MAX_K:
        raise ValueError(f"Codebook size must be between 1 and {MAX_K}.")
    check_outputs([cb_file, bin_file], [image])
    H, W = load_gray(image).shape   # also fails early if this is not an image

    start = time.time()
    compress(image, h, w, K, cb_file, bin_file)
    seconds = time.time() - start

    actual_K = read_header(cb_file, bin_file)[4]
    if actual_K != K:
        reason = "rounded up to a power of 2"
        if actual_K < next_pow2(K):
            reason = "reduced because the image has too few blocks"
        print(f"Note: codebook size changed from {K} to {actual_K} ({reason}).")

    print(f"\nImage: {W} x {H} (grayscale), block {h} x {w}, codebook size {actual_K}")
    print(f"Saved codebook         : {cb_file} ({os.path.getsize(cb_file)} bytes)")
    print(f"Saved compressed stream: {bin_file} ({os.path.getsize(bin_file)} bytes)")
    # original, compressed, ratio = file_ratio(H, W, cb_file, bin_file)
    # print(f"Actual ratio      : {original} / {compressed} bytes = {ratio:.2f}:1")
    original, labels, codebook, ratio = theoretical_ratio(H, W, h, w, actual_K)
    print(f"Theoretical ratio : {original} / ({labels} label bits + {codebook} codebook bits)"
          f" = {ratio:.2f}:1")
    print(f"Time: {seconds:.2f} s")


def run_decompress(cb_file, bin_file, out_path, original=None):
    require_file(cb_file)
    require_file(bin_file)
    if not out_path.lower().endswith(LOSSLESS):
        raise ValueError(f"Output image must be lossless: {', '.join(LOSSLESS)}")
    inputs = [cb_file, bin_file]
    if original:
        require_file(original)
        inputs.append(original)
    check_outputs([out_path], inputs)
    H, W, h, w, K = read_header(cb_file, bin_file)

    decompress(cb_file, bin_file, out_path)
    print(f"\nSaved reconstructed image: {out_path} ({W} x {H})")
    # ratio = file_ratio(H, W, cb_file, bin_file)[2]
    # print(f"Actual compression ratio : {ratio:.2f}:1")

    if original:
        o, r = load_gray(original), load_gray(out_path)
        if o.shape != r.shape:
            raise ValueError("The original image has a different size, so MSE cannot be computed.")
        error = mse(o, r)
        print(f"MSE : {error:.4f}")
        # print(f"PSNR: {psnr(error):.2f} dB")


# ---------- Interactive menu ----------
def ask(prompt, default=None):
    shown = f" [{default}]" if default is not None else ""
    # strip quotes: dragging a file into the terminal often adds them
    answer = input(f"{prompt}{shown}: ").strip().strip('"\'')
    if answer:
        return answer
    return "" if default is None else str(default)


def ask_int(prompt, default):
    while True:
        try:
            return int(ask(prompt, default))
        except ValueError:
            print("Please enter a whole number.")


def menu():
    while True:
        print("\n===== Vector Quantization =====")
        print("1) Compress\n2) Decompress\n3) Exit")
        try:
            choice = ask("Choose")
            if choice == '1':
                run_compress(ask("Image path"),
                             ask_int("Block height", 4), ask_int("Block width", 4),
                             ask_int("Codebook size", 256),
                             ask("Codebook output", "codebook.bin"),
                             ask("Compressed output", "compressed.bin"))
            elif choice == '2':
                run_decompress(ask("Codebook file", "codebook.bin"),
                               ask("Compressed file", "compressed.bin"),
                               ask("Output image", "reconstructed.png"),
                               ask("Original image for MSE (Enter to skip)") or None)
            elif choice == '3':
                return
            else:
                print("Enter 1, 2 or 3.")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except (ValueError, OSError) as e:
            print("Error:", e)


# ---------- Command line ----------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:          # no arguments => friendly menu for the TA
        menu()
        return 0

    p = argparse.ArgumentParser(description="Vector Quantization image compressor "
                                            "(run without arguments for a menu)")
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("compress")
    c.add_argument("image")
    c.add_argument("--h", type=int, default=4, help="block height")
    # "-h" is --help, while "--h" is the block height
    c.add_argument("--w", type=int, default=4, help="block width")
    c.add_argument("--k", type=int, default=256, help="codebook size")
    c.add_argument("--cb", default="codebook.bin")
    c.add_argument("--out", default="compressed.bin")

    d = sub.add_parser("decompress")
    d.add_argument("--cb", default="codebook.bin")
    d.add_argument("--inp", default="compressed.bin")
    d.add_argument("--out", default="reconstructed.png")
    # d.add_argument("--orig", help="original image, to print MSE and PSNR")
    d.add_argument("--orig", help="original image, to print MSE")

    a = p.parse_args(argv)
    try:
        if a.mode == "compress":
            run_compress(a.image, a.h, a.w, a.k, a.cb, a.out)
        else:
            run_decompress(a.cb, a.inp, a.out, a.orig)
    except (ValueError, OSError) as e:
        print("Error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())