import struct
import numpy as np
from PIL import Image

# data => the raw bytes read from the compressed file (everything after the 8-byte header).
# count => the number of labels to recover, which equals the number of blocks in the image.
# bits => how many bits each label uses (3 for K=8)
def unpack_indices(data, count, bits):
    bit_arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:count * bits]
    # np.frombuffer: reads the bytes as an array of numbers from 0 to 255.
    # np.unpackbits: turns each byte into its 8 individual bits, so the result is a long array of 0s and 1s.
    # [:count * bits] keeps only the real bits. The compressor padded the last byte with zeros, and this cuts that padding off.
    bit_arr = bit_arr.reshape(count, bits)
    # bit_arr: has one row per label and one column per bit. With 3 bits, one row looks like [0, 1, 1]
    weights = 1 << np.arange(bits - 1, -1, -1)
    # generates numbers in descending order from bits - 1 "2" until > -1, and shifts 1 n times to get
    # 1 << n = 2^n. Final weights holds [4, 2, 1]
    return (bit_arr * weights).sum(axis=1)
    # multiplies each bit by its weight, add each row to get its label, returns an array of labels at the end
    # [0, 1, 1] gives [0, 2, 1] adding them gives 3 which is the label of that block  



# blocks => the table of codewords, one row per block, each row holding h*w pixel values.
# Hp, Wp => the height and width of the padded image.
# h, w => the block height and width.
def blocks_to_image(blocks, Hp, Wp, h, w):
    img = np.zeros((Hp, Wp)) # empty canvas of the padded size that gets filled in 
    n = 0 # counter of blocks, start at top-left 
    for i in range(0, Hp, h): # (start, stop, step)
        for j in range(0, Wp, w):
            img[i:i+h, j:j+w] = blocks[n].reshape(h, w)
            # Take flat block number n, convert it into an h × w matrix, and insert it into the image at the correct position
            n += 1
    return img


def decompress(cb_file, bin_file, out_path):
    with open(cb_file, 'rb') as f:
        h, w, K = struct.unpack('<BBH', f.read(4))
        # 4 bytes header, h = 1, w = 1, k = 2 
        cb = np.frombuffer(f.read(K * h * w), dtype=np.uint8).reshape(K, h * w)
        # f.read(K * h * w) reads exactly the number of bytes the codewords take up.  
        # This is why K, h and w had to be read first: they say how many bytes to read.
        # np.frombuffer(..., dtype=np.uint8) turns those raw bytes into a NumPy array of numbers, one number per byte, each from 0 to 255.
        # .reshape(K, h * w) arranges the flat list into a table with K rows and h*w columns. In the example that is 4 rows of 4 values. Row 0 is codeword 0, row 1 is codeword 1, and so on. Each row is one flattened block, in the same order image_to_blocks flattened it.

    with open(bin_file, 'rb') as f:
        H, W = struct.unpack('<II', f.read(8))
        data = f.read() # compressed file data

    Hp, Wp = H + (-H) % h, W + (-W) % w # Hp and Wp are the padded height and width 
    count = (Hp // h) * (Wp // w) # number of blocks
    bits = max(1, int(np.ceil(np.log2(K))))
    # number of bits per label. np.log2(K) gives 3 for K=8, ceil rounds up in case K is not a power of 2, and max(1, ...) 
    # makes sure it is at least 1 when K=1.

    idx = unpack_indices(data, count, bits) # array of labels recovered from the bit stream
    img = blocks_to_image(cb[idx], Hp, Wp, h, w)[:H, :W]
    # cb[idx] looks up the codeword for every label, which gives one block per label.
    # blocks_to_image(...) assembles those blocks into the padded image.
    # [:H, :W] crops off the padding, so img ends up with the original size
    Image.fromarray(img.astype(np.uint8)).save(out_path)
    # img.astype(np.uint8) converts the values to 8-bit pixels, and Image.fromarray(...).save(out_path) writes the image file