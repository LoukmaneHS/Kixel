from ..classes.karacter import Karacter
from ..classes.kmatrix import Kmatrix
from ..classes.kimage import Kimage
from ..classes.kframe import Kframe


def create(kind, *args, **kwargs):
    if kind is Karacter:
        return Karacter(*args, **kwargs)
    elif kind is Kframe:
        return Kframe(*args, **kwargs)
    elif kind is Kimage:
        return Kimage(*args, **kwargs)
    elif kind is Kmatrix:
        return Kmatrix(*args, **kwargs)
    else:
        raise TypeError(f"create() does not support kind {kind!r}")
