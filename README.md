Kixel

«Lossless image representation for kinematic motion data.»

Kixel is a Python library for representing kinematic motion as images without losing numerical precision.

The library converts motion data from robots, articulated rigs, or 3D characters into an RGBA image representation that preserves every original bit of information. This allows motion sequences to be processed using image-oriented tools and machine learning pipelines while remaining fully reversible.

Unlike traditional approaches that normalize values into a limited image range, Kixel stores each motion value as a full "int32" and encodes its four bytes directly into the RGBA channels of a pixel.

Motivation

Modern computer vision architectures such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) are optimized for image-like inputs.

Kixel explores a simple question:

«Can kinematic motion be represented as an image while preserving the exact original data?»

To answer this, Kixel treats each motion value as a collection of four bytes and stores those bytes directly in RGBA space. The resulting image can be consumed by image-based pipelines while remaining a lossless representation of the underlying motion.

Architecture

Karacter
    │
    ├── create_motion()
    ▼
Kmatrix (frames × dof)
    │
    ├── encode_image()
    ▼
Kimage (frames × dof × 4)
    │
    ├── decode_image()
    ▼
Kmatrix
    │
    ├── matframe()
    └── imgframe()
            ▼
         Kframe
            │
            └── update_accumulator()
                    ▼
                Karacter

Data Model

Karacter

Represents a kinematic system such as a robot, articulated rig, or animated character.

Attributes:

Attribute| Type| Description
"model_name"| "str"| Human-readable identifier
"dof_count"| "uint16"| Number of degrees of freedom
"accumulator"| "uint32[dof]"| Current accumulated state of every DOF

---

Kmatrix

Stores motion as a row-major matrix of signed 32-bit integers.

Shape:

(frames, dof)

Where:

- Each row represents a frame.
- Each column represents a degree of freedom.
- Values are motion deltas encoded as "int32".

---

Kimage

Lossless RGBA representation of a "Kmatrix".

Shape:

(frames, dof, 4)

The final dimension contains:

[R, G, B, A]

corresponding to the four bytes of an "int32" value.

---

Kframe

Represents a single motion frame.

Shape:

(dof,)

Implemented as a direct subclass of "numpy.ndarray" for seamless interoperability with NumPy operations.

Encoding Strategy

Each motion value is stored as a signed 32-bit integer:

int32

Kixel converts the value into four bytes using a fixed big-endian layout:

int32
  │
  ▼
[B0][B1][B2][B3]
  │   │   │   │
  ▼   ▼   ▼   ▼
  R   G   B   A

The reverse operation reconstructs the original integer exactly.

Because the transformation operates directly on raw bytes:

- No normalization is performed.
- No quantization is performed.
- No rounding occurs.
- No precision is lost.

Accumulator Model

Motion frames store relative changes (deltas).

The accumulator stores the current absolute state.

For each frame:

accumulator ← accumulator + delta

Internally:

- Frame values are represented as "int32".
- Accumulator values are represented as "uint32".
- Arithmetic wraps naturally at "2³²".

This provides a compact circular representation suitable for rotational systems.

Lossless Guarantee

The following identity always holds:

decoded = decode_image(encode_image(kmatrix))

np.array_equal(
    kmatrix.kmatrix,
    decoded.kmatrix
)
# True

Encoding and decoding preserve every bit of the original motion data.

Example

from kixel import (
    Karacter,
    create_motion,
    encode_image,
    decode_image,
    matframe,
    update_accumulator,
)

robot = Karacter("robot_arm", 6)

motion = create_motion(
    karacter=robot,
    frames_number=100
)

image = encode_image(motion)

restored = decode_image(image)

frame = matframe(0, motion)

update_accumulator(robot, frame)

Byte Order

Kixel uses a fixed big-endian representation for all encoding and decoding operations.

This ensures identical results across platforms regardless of native machine endianness.

Design Goals

- Lossless motion representation
- Deterministic encoding and decoding
- NumPy-first implementation
- Explicit memory layout
- Platform-independent byte ordering
- Compatibility with image-processing workflows

Current Status

Version: "0.0.1"

Implemented features:

- "Karacter"
- "Kmatrix"
- "Kimage"
- "Kframe"
- Motion creation
- Frame extraction
- Lossless image encoding
- Lossless image decoding
- Accumulator updates

Future Directions

Potential areas of exploration include:

- Dataset generation utilities
- PyTorch integration
- TensorFlow integration
- Motion visualization tools
- Compression experiments
- Temporal batching utilities
- Research workflows for vision-based motion understanding

License

License information will be added in a future release.
