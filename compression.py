import struct
import numpy as np
from PIL import Image

# h, w => height and width of the block
# H, W => height and width of the image

# ---------- Padding ----------
def pad_image(img, h, w):
    H, W = img.shape
    pad_rows = (-H) % h # how many rows are left to make H % h =0
    pad_cols = (-W) % w # how many columns are left to make W % w =0
    return np.pad(img, ((0, pad_rows), (0, pad_cols)), mode='edge')
    # np.pad(): adds padding around an image
    # (top, bottom), (left, right). put the padding rows at bottom and right. 
    # mode = 'edge': copy the pixels at the corners to pad   


# ---------- Image -> vectors ----------
def image_to_blocks(img, h, w):
    H, W = img.shape
    blocks = []
    for i in range(0, H, h):
        for j in range(0, W, w):
            block = img[i:i+h, j:j+w].flatten()
            blocks.append(block)
    return np.array(blocks, dtype=float)


# ---------- Distance + nearest codeword ----------
def distances_to_codebook(vectors, codebook):
    dist = np.zeros((len(vectors), len(codebook)))
    for k in range(len(codebook)):
        dist[:, k] = np.abs(vectors - codebook[k]).sum(axis=1)
    return dist


def get_indices(vectors, codebook):
    return np.argmin(distances_to_codebook(vectors, codebook), axis=1)


# ---------- LBG with splitting ----------
def next_pow2(K): 
    return 1 << (K - 1).bit_length()          # 6 -> 8, 8 -> 8


def lbg(vectors, K, eps=1.0, max_iter=100):
    K = next_pow2(K)
    while K > len(vectors):
        K //= 2

    codebook = vectors.mean(axis=0, keepdims=True)

    while len(codebook) < K:
        codebook = np.concatenate([codebook - eps, codebook + eps])
        codebook = refine(vectors, codebook, max_iter)

    return np.clip(np.rint(codebook), 0, 255)


def refine(vectors, codebook, max_iter):
    old_labels = None
    for _ in range(max_iter):
        dist = distances_to_codebook(vectors, codebook)
        labels = np.argmin(dist, axis=1)

        if old_labels is not None and np.array_equal(labels, old_labels):
            break
        old_labels = labels

        for k in range(len(codebook)):
            members = vectors[labels == k]
            if len(members) > 0:
                codebook[k] = members.mean(axis=0)
            else:
                worst = np.argmax(dist[np.arange(len(vectors)), labels])
                codebook[k] = vectors[worst]
    return codebook


# ---------- Bit packing ----------
def pack_indices(idx, bits):
    shifts = np.arange(bits - 1, -1, -1)
    bit_arr = ((idx[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    return np.packbits(bit_arr).tobytes()


# ---------- Compress ----------
def compress(img_path, h, w, K, cb_file, bin_file):
    img = np.array(Image.open(img_path).convert('L'))
    H, W = img.shape
    vecs = image_to_blocks(pad_image(img, h, w), h, w)

    cb = lbg(vecs, K)
    K = len(cb)                               # actual K (power of 2)
    idx = get_indices(vecs, cb)
    bits = max(1, int(np.log2(K)))

    with open(cb_file, 'wb') as f:            # codebook file
        f.write(struct.pack('<BBH', h, w, K))
        f.write(cb.astype(np.uint8).tobytes())
    with open(bin_file, 'wb') as f:           # compressed binary stream
        f.write(struct.pack('<II', H, W))
        f.write(pack_indices(idx, bits))