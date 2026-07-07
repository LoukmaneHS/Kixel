import numpy as np
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage


def decode(kimage: Kimage) -> Kmatrix:
    row = int(kimage.row)
    column = int(kimage.column)
    raw = kimage.kimage.reshape(row, column, 4).view('>i4').reshape(row, column)
    mat = Kmatrix(row=row, column=column)
    mat.kmatrix[:] = raw.astype(np.int32)
    return mat
