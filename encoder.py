import numpy as np
#padding 
def pad_image(img, h , w):
    H, W = img.shape
    pad_rows = (-H) % h
    pad_cols = (-W) % w
    return np.pad(img , ((0, pad_rows), (0, pad_cols)), mode='edge')

#cut image into blocks left to right, top to bottom
def image_to_blocks(img, h, w):
    H, W = img.shape
    blocks = []
    for i in range(0, H, h):
        for j in range(0, W, w):
            block = img[i:i+h, j:j+w].flatten()
            blocks.append(block)
    return np.array(blocks, dtype=float)

# Distance + nearest codeword
def distances_to_codebook(vectors, codebook):
    dist = np.zeros(((len(vectors), len(codebook))))
    for k in range(len(codebook)):
        dist[:, k] = np.abs(vectors - codebook[k]).sum(axis=1)
    return dist
    
def get_indices(vectors, codebook):
    return np.argmin(distances_to_codebook(vectors, codebook), axis=1)
  
    
#  LBG with splitting    
def lbg(vectors, K, eps=1.0, max_iter=100):
    codebook = vectors.mean(axis=0, keepdims=True)

    while len(codebook) < K:
        if 2 * len(codebook) <= K:
            to_split = set(range(len(codebook)))          
        else:
            labels = get_indices(vectors, codebook)
            counts = np.bincount(labels, minlength=len(codebook))
            need = K - len(codebook)
            to_split = set(np.argsort(-counts)[:need])

        new_cb = []
        for i, c in enumerate(codebook):
            if i in to_split:
                new_cb.append(c - eps)        
                new_cb.append(c + eps)        
            else:
                new_cb.append(c)
        codebook = np.array(new_cb)

        codebook = refine(vectors, codebook, 1)

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








# Quick test on the lecture example (6x6, block 2x2, K=4)
# ---------------------------------------------------------------
# if __name__ == "__main__":
#     img = np.array([
#         [1, 2, 7, 9, 4, 11],
#         [3, 4, 6, 6, 12, 12],
#         [4, 9, 15, 14, 9, 9],
#         [10, 10, 20, 18, 8, 8],
#         [4, 3, 17, 16, 1, 4],
#         [4, 5, 18, 18, 5, 6]])
#     img = pad_image(img, 2, 2)
#     vecs = image_to_blocks(img, 2, 2)
#     cb = lbg(vecs, 4)
#     print("Codebook:\n", cb)
#     print("Indices:", get_indices(vecs, cb))