from ..classes.karacter import Karacter
from ..classes.kinematrix import Kinematrix


def create_motion(karacter: Karacter, frames_number: int) -> Kinematrix:
    return Kinematrix(column=int(karacter.dof_count), row=frames_number)
