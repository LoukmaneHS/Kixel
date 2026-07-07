import numpy as np


class Frame(np.ndarray):
    def __new__(cls, dof: int):
        if not (0 <= dof <= 65535):
            raise ValueError(f"dof must be between 0 and 65535, got {dof}")

        obj = np.zeros(dof, dtype=np.int32).view(cls)
        obj.dof_count = np.uint16(dof)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dof_count = getattr(obj, 'dof_count', None)

    def __repr__(self):
        return f"Frame(dof={self.dof_count}, values={np.asarray(self)})"
