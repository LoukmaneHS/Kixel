import numpy as np
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage
from ..classes.kframe import Kframe


def flow(source, index: np.uint16) -> Kframe:
    if isinstance(source, Kmatrix):
        max_index = source.kmatrix.shape[0]
    elif isinstance(source, Kimage):
        max_index = source.kimage.shape[0]
    else:
        raise TypeError(f"flow() does not support source of type {type(source).__name__}")

    if not (0 <= index < max_index):
        raise IndexError(f"index must be between 0 and {max_index - 1}, got {index}")

    if isinstance(source, Kmatrix):
        dof = source.kmatrix.shape[1]
        frame = Kframe(dof=dof)
        frame[:] = source.kmatrix[index]
        return frame

    dof = source.kimage.shape[1]
    raw_row = source.kimage[index].reshape(dof, 4).view('>i4').reshape(dof)
    frame = Kframe(dof=dof)
    frame[:] = raw_row.astype(np.int32)
    return frame
