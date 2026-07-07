import numpy as np
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage
from ..classes.kframe import Kframe


def create_frame(dof: int) -> Kframe:
    return Kframe(dof=dof)


def matframe(index: np.uint64, kmatrix: Kmatrix) -> Kframe:
    max_index = kmatrix.kinematrix.shape[0]

    if not (0 <= index < max_index):
        raise IndexError(f"index must be between 0 and {max_index - 1}, got {index}")

    dof = kmatrix.kinematrix.shape[1]
    frame = Kframe(dof=dof)
    frame[:] = kmatrix.kinematrix[index]

    return frame


def imgframe(index: np.uint64, kimage: Kimage) -> Kframe:
    max_index = kimage.kimage.shape[0]

    if not (0 <= index < max_index):
        raise IndexError(f"index must be between 0 and {max_index - 1}, got {index}")

    dof = kimage.kimage.shape[1]

    raw_row = kimage.kimage[index].reshape(dof, 4).view('>i4').reshape(dof)

    frame = Kframe(dof=dof)
    frame[:] = raw_row.astype(np.int32)

    return frame
