import numpy as np

class Karacter:
    def __init__(self, name: str, dof: int):
        if not (0 <= dof <= 65535):
            raise ValueError(f"dof must be between 0 and 65535, got {dof}")
        
        self.model_name = name
        self.dof_count = np.uint16(dof)
        self.asa = np.zeros(dof, dtype=np.uint16)

    def __repr__(self):
        return f"Karacter(name={self.model_name!r}, dof={self.dof_count})"
