import numpy as np
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage


def encode(kmatrix: Kmatrix) -> Kimage:
    img = Kimage(row=int(kmatrix.row), column=int(kmatrix.column))
    raw = kmatrix.kmatrix.astype('>i4')
    byte_view = raw.view(np.uint8).reshape(raw.shape[0], raw.shape[1], 4)
    img.kimage[:] = byte_view
    return img
