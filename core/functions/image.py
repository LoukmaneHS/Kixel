import numpy as np
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage


def encode_image(kmatrix: Kmatrix) -> Kimage:
    img = Kimage(row=int(kmatrix.row), column=int(kmatrix.column))

    raw = kmatrix.kmatrix.astype('>i4')
    byte_view = raw.view(np.uint8).reshape(raw.shape[0], raw.shape[1], 4)

    img.kimage[:] = byte_view

    return img


def decode_image(kimage: Kimage) -> Kmatrix:
    row = int(kimage.row)
    column = int(kimage.column)

    raw = kimage.kimage.reshape(row, column, 4).view('>i4').reshape(row, column)

    mat = Kmatrix(row=row, column=column)
    mat.kmatrix[:] = raw.astype(np.int32)

    return mat
