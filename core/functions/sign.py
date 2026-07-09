import numpy as np
from ..classes.karacter import Karacter
from ..classes.ksign import Ksign


def sign(karacter: Karacter, ksign: Ksign) -> None:
    if int(karacter.dof_count) != int(ksign.dof_count):
        raise ValueError(
            f"karacter dof ({karacter.dof_count}) does not match "
            f"ksign dof_count ({ksign.dof_count})"
        )

    diff = karacter.n_acc.astype(np.int64) - karacter.o_acc.astype(np.int64)
    ksign.values[:] = diff >= 0
