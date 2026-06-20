import numpy as np
import cv2
from numpy.linalg import svd as _svd
from pywt import dwt2, idwt2
from cv2 import dct, idct


def _permutations(seed, n, k):
    return np.random.RandomState(seed).random(size=(n, k)).argsort(axis=1)


def _embed_block(block, shuffler, bit, d1, d2):
    coeff = dct(block)
    flat = coeff.flatten()
    shuffled = flat[shuffler].reshape(4, 4)
    u, s, vh = _svd(shuffled)
    s[0] = (s[0] // d1 + 0.25 + 0.5 * bit) * d1
    if d2:
        s[1] = (s[1] // d2 + 0.25 + 0.5 * bit) * d2
    recon = (u @ np.diag(s) @ vh).flatten()
    flat[shuffler] = recon
    return idct(flat.reshape(4, 4))


def erase(img, d1=36, d2=20):
    h, w = img.shape[:2]
    img_f32 = img.astype(np.float32)
    img_yuv = cv2.cvtColor(img_f32, cv2.COLOR_BGR2YUV)

    uh = img_yuv.shape[0] - img_yuv.shape[0] % 8
    uw = img_yuv.shape[1] - img_yuv.shape[1] % 8

    pw_img = np.random.randint(1, 2 ** 16)
    pw_wm = np.random.randint(1, 2 ** 16)

    out_yuv = img_yuv.copy()

    for ch in range(3):
        ca, *hvd = dwt2(img_yuv[:uh, :uw, ch], 'haar')
        ca = ca.astype(np.float32, copy=True)
        ca_h, ca_w = ca.shape
        nbh = ca_h // 4
        nbw = ca_w // 4

        strides = (ca_w * 4 * 4, 4 * 4, ca_w * 4, 4)
        blocks = np.lib.stride_tricks.as_strided(
            ca[:nbh * 4, :nbw * 4], (nbh, nbw, 4, 4), strides
        )

        n = nbh * nbw
        perms = _permutations(pw_img + ch, n, 16)
        rng = np.random.RandomState(pw_wm + ch)
        bits = rng.randint(0, 2, size=n)

        for i in range(nbh):
            for j in range(nbw):
                blk = blocks[i, j]
                blocks[i, j] = _embed_block(
                    blk, perms[i * nbw + j], bits[i * nbw + j], d1, d2
                )

        out_yuv[:uh, :uw, ch] = idwt2((ca, *hvd), 'haar')

    out_yuv = out_yuv[:h, :w]
    out = cv2.cvtColor(out_yuv, cv2.COLOR_YUV2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)
