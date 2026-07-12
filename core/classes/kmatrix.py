import numpy as np

class Kmatrix:
    def __init__(self, row: int, column: int):
        if not (0 <= row <= 65535):
            raise ValueError(f"row must be between 0 and 65535, got {row}")
        if not (0 <= column <= 65535):
            raise ValueError(f"column must be between 0 and 65535, got {column}")

        self.row = np.uint16(row)
        self.column = np.uint16(column)
        self.kmatrix = np.zeros((row, column), dtype=np.int32, order='C')

    def __repr__(self):
        return f"Kmatrix(row={self.row}, column={self.column})"
