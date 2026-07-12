import numpy as np
from ..classes.karacter import Karacter


def sign(karacter: Karacter) -> None:
    diff = (karacter.nacc - karacter.oacc).view(np.int32)
    karacter.sacc[:] = diff >= 0
