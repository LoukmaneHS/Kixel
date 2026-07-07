import numpy as np
from ..classes.karacter import Karacter


def update_accumulator(karacter: Karacter, kframe: Kframe) -> None:
    if kframe.shape[0] != int(karacter.dof_count):
        raise ValueError(
            f"kframe dof ({kframe.shape[0]}) does not match "
            f"karacter dof_count ({karacter.dof_count})"
        )

    delta_bits = kframe.view(np.uint32)
    karacter.accumulator = (karacter.accumulator + delta_bits).astype(np.uint32)

    kframe[:] = 0


def acc_reset(karacter: Karacter) -> None:
    karacter.accumulator.fill(0)


def acc_state(karacter: Karacter) -> np.ndarray:
    return karacter.accumulator.copy()


def kar_rename(karacter: Karacter, new_name: str) -> None:
    if not new_name or not isinstance(new_name, str):
        raise ValueError("Name must be a non-empty string")
    karacter.model_name = new_name


def kar_info(karacter: Karacter) -> None:
    print(f"Karacter: {karacter.model_name}")
    print(f"  DOF   : {karacter.dof_count}")
    print(f"  Acc   : shape={karacter.accumulator.shape}, "
          f"dtype={karacter.accumulator.dtype}")
