import numpy as np
from ..classes.kinematrix import Kinematrix
from ..classes.kimage import Kimage


def encoding_image(kinematrix: Kinematrix, big_endian: bool = True) -> Kimage:
    img = Kimage(row=int(kinematrix.row), column=int(kinematrix.column))

    dtype_str = '>i4' if big_endian else '<i4'
    raw = kinematrix.kinematrix.astype(dtype_str)

    byte_view = raw.view(np.uint8).reshape(raw.shape[0], raw.shape[1], 4)

    img.kimage[:] = byte_view

    return img
