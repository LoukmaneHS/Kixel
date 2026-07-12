import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.classes.karacter import Karacter
from core.classes.kmatrix import Kmatrix
from core.functions.create import create
from core.functions.flow import flow
from core.functions.encode import encode
from core.functions.decode import decode
from core.functions.update import update
from core.functions.sign import sign
from core.functions.state import state
from core.functions.clear import clear


def main() -> None:
    DOF = 6
    FRAMES = 4

    karacter: Karacter = create(Karacter, dof=DOF, name="TestRig")
    print(karacter)

    kmatrix: Kmatrix = create(Kmatrix, row=FRAMES, column=DOF)

    sample_motion = np.array(
        [
            [   10,   -5,    0,   3, -1000,      7],
            [   -3,    8, 1200,   0,     4, -99999],
            [2**31 - 1, -2**31, 0, 42,   -42,      0],
            [    1,     1,    1,  1,     1,      1],
        ],
        dtype=np.int32,
    )
    kmatrix.kmatrix[:] = sample_motion

    kimage = encode(kmatrix)
    print(f"Encoded {kmatrix} -> {kimage}")

    decoded = decode(kimage)
    assert np.array_equal(kmatrix.kmatrix, decoded.kmatrix), "Round-trip lost data!"
    print("Lossless round-trip verified: decode(encode(kmatrix)) == kmatrix")

    for i in range(FRAMES):
        frame = flow(np.uint16(i), kmatrix)
        update(karacter, frame)
        sign(karacter)

    print("Final accumulator state (nacc):", state(karacter))
    print("Sign flags (sacc):", karacter.sacc)

    clear(karacter)
    clear(kmatrix)
    clear(kimage)
    print("Cleared karacter, kmatrix, and kimage.")


if __name__ == "__main__":
    main()