from ..classes.karacter import Karacter
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage
from ..classes.kframe import Kframe


def clear(obj) -> None:
    if isinstance(obj, Kframe):
        obj[:] = 0
    elif isinstance(obj, Kmatrix):
        obj.kmatrix[:] = 0
    elif isinstance(obj, Kimage):
        obj.kimage[:] = 0
    elif isinstance(obj, Karacter):
        obj.nacc[:] = 0
        obj.oacc[:] = 0
        obj.sacc[:] = False
    else:
        raise TypeError(f"clear() does not support objects of type {type(obj).__name__}")
