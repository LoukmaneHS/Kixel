import numpy as np
from ..classes.karacter import Karacter
from ..classes.kframe import Kframe


def update(karacter: Karacter, kframe: Kframe) -> None:
    if kframe.shape[0] != int(karacter.dof_count):
        raise ValueError(
            f"kframe dof ({kframe.shape[0]}) does not match "
            f"karacter dof_count ({karacter.dof_count})"
        )
    
    karacter.o_acc[:] = karacter.n_acc
    delta_bits = kframe.view(np.uint32)
    karacter.n_acc = (karacter.n_acc + delta_bits).astype(np.uint32)
