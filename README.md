# Kixel `0.0.1`

Kixel is a small Python library that converts the motion of a 3D model or robot (a *Kinematic Character*) into a lossless image representation, so that vision models (ViT/CNN) can be trained to understand **motion** instead of ordinary images.

Instead of normalizing angle values into a 0–255 range (which loses precision), Kixel encodes each `int32` motion value directly into the four RGBA channels of a pixel, one byte per channel — so no information is lost between the raw motion data and its image form. The transformation is fully reversible.

## Core Idea

```
Karacter ──(create_motion)──► Kmatrix ──(encode_image)──► Kimage
    ▲                             │                            │
    │                             ├──(matframe)──► Kframe ◄─────┤(imgframe)
    │                             │                            
    │                             └──(decode_image)◄───────────┘
    │
    └────────────(update_accumulator)◄──────────────┘
```

A `Karacter`'s motion over time is stored as a `Kmatrix` (one row per frame, one column per degree of freedom). That matrix can be losslessly encoded into a `Kimage` (RGBA), and either representation can be decoded back or queried for a single `Kframe` at a given index. Each `Kframe` holds a *delta* (`d`) per DOF, which `update_accumulator` accumulates into the `Karacter`'s absolute angle — driving its actual motion.

## Classes

| Class | Description |
|---|---|
| `Karacter` | Represents a kinematic model (robot / rig / 3D character). Holds a `model_name`, a `dof_count` (`uint16`, number of degrees of freedom), and an `accumulator` (`uint32`) that holds the current absolute angle of each DOF, wrapping naturally at `360° = 0°`. |
| `Kmatrix` | A row-major `int32` matrix of shape `(frames, dof)`. Each row is one full frame, stored contiguously in memory for fast GPU access. |
| `Kimage` | A row-major `uint8` matrix of shape `(row, column, 4)` — an RGBA image representation of a `Kmatrix`. |
| `Kframe` | A single frame: an `int32` array of length `dof`, subclassing `np.ndarray` directly so it behaves like a plain array. |

## Functions

### `motion.py`
- **`create_motion(karacter: Karacter, frames_number: int) -> Kmatrix`**
  Builds an empty `Kmatrix` sized for the character's DOF count and the requested number of frames.

### `image.py`
- **`encode_image(kmatrix: Kmatrix) -> Kimage`**
  Encodes a `Kmatrix` into a `Kimage` by splitting each `int32` value into 4 bytes (big-endian) mapped to the R, G, B, A channels.
- **`decode_image(kimage: Kimage) -> Kmatrix`**
  Reverses `encode_image`, reconstructing the original `Kmatrix` from a `Kimage`.

### `frame.py`
- **`create_frame(dof: int) -> Kframe`**
  Creates an empty `Kframe` of the given length.
- **`matframe(index: np.uint64, kmatrix: Kmatrix) -> Kframe`**
  Extracts a single frame from a `Kmatrix` by row index.
- **`imgframe(index: np.uint64, kimage: Kimage) -> Kframe`**
  Extracts a single frame directly from a `Kimage` by decoding one row.

### `accumulator.py`
- **`update_accumulator(karacter: Karacter, kframe: Kframe) -> None`**
  Adds each delta value (`d`) held in a `Kframe` to the matching DOF's absolute angle in the `Karacter`'s `accumulator`, then zeroes out the `Kframe`. The addition wraps automatically at `2³²`, mirroring the `360° = 0°` circularity of the angle:
  - Accumulator step: `360 / 4294967295` (`360 / 2³²`)
  - Frame (delta) step: `180 / 2147483648` (`180 / 2³¹`, equal to `360 / 2³²`)

  Because both steps are identical, the raw bits of the `int32` delta can be added directly onto the `uint32` accumulator with no rescaling.

## Example

```python
from kixel import Karacter, create_motion, encode_image, decode_image, matframe, update_accumulator

robot = Karacter("robot_arm", 6)
motion = create_motion(robot, 100)          # Kmatrix: 100 frames x 6 DOF

image = encode_image(motion)                # Kimage: lossless RGBA encoding
restored = decode_image(image)              # Kmatrix: reconstructed, identical to `motion`

frame = matframe(0, motion)                 # Kframe: deltas for frame 0
update_accumulator(robot, frame)            # advances robot.accumulator by that frame's deltas
```

## Byte Order

All encoding/decoding uses a fixed **big-endian** byte order. This guarantees identical results across machines regardless of native endianness, and is not currently configurable.

## Status

`0.0.1` — first working version. The full pipeline is in place: building motion containers, lossless encoding/decoding between `Kmatrix` and `Kimage`, extracting individual frames from either representation, and accumulating frame deltas into a `Karacter`'s live angular state.
