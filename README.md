# 🎬 Kixel

> **Lossless image representation for kinematic motion data.**

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg) ![Version](https://img.shields.io/badge/version-0.0.2-orange.svg) ![License](https://img.shields.io/badge/license-MPL--2.0-green.svg)

**Kixel** is a Python library for representing kinematic motion as images without losing numerical precision.

The library converts motion data from robots, articulated rigs, or 3D characters into an RGBA image representation that preserves every original bit of information. This allows motion sequences to be processed using image-oriented tools and machine learning pipelines while remaining fully reversible.

Unlike traditional approaches that normalize values into a limited image range, Kixel stores each motion value as a full `int32` and encodes its four bytes directly into the RGBA channels of a pixel.

---

## 📑 Table of Contents
- [Motivation](#-motivation)
- [Design Philosophy](#-design-philosophy)
- [Architecture](#-architecture)
- [Data Model](#-data-model)
- [Core Functions](#-core-functions)
- [Encoding Strategy](#-encoding-strategy)
- [Accumulator Model](#-accumulator-model)
- [Lossless Guarantee](#-lossless-guarantee)
- [Usage Example](#-usage-example)
- [PNG Integration](#-png-integration)
- [Design Goals](#-design-goals)
- [Status & Future](#-status--future-directions)
- [License](#-license)

---

## 💡 Motivation

Modern computer vision architectures such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) are optimized for image-like inputs.

Kixel explores a simple question:

> *Can kinematic motion be represented as an image while preserving the exact original data?*

To answer this, Kixel treats each motion value as a collection of four bytes and stores those bytes directly in RGBA space. The resulting image can be consumed by image-based pipelines while remaining a **100% lossless** representation of the underlying motion.

---

## 🧭 Design Philosophy

Kixel is **not** an object-oriented API. Instead of calling methods on objects (`robot.update(frame)`), Kixel exposes a small set of **generic verbs** that dispatch based on the type of object passed to them:

```python
create(Karacter, 6)     # instead of Karacter(6)
clear(frame)            # works on Kframe, Kmatrix, Kimage, or Karacter
flow(0, motion)         # works on Kmatrix or Kimage
info(robot)             # works on any of the 4 core types
```

This keeps the vocabulary small and consistent, no matter how many data types or files the library grows into. Only **4 core data types** exist (`Karacter`, `Kframe`, `Kmatrix`, `Kimage`), and every generic function is written to support all of them explicitly.

---

## 🏗️ Architecture

```text
  ┌────────────┐
  │  Karacter  │
  └────────────┘

  ┌────────────┐
  │  Kmatrix   │  (frames × dof)
  └─────┬──────┘
        │ encode()
        ▼
  ┌────────────┐
  │   Kimage   │  (frames × dof × 4)
  └─────┬──────┘
        │ decode()
        ▼
  ┌────────────┐      ┌────────────┐
  │  Kmatrix   │─────▶│   Kframe   │  (flow)
  └────────────┘      └─────┬──────┘
                            │ update() / updatc()
                            ▼
                      ┌────────────┐
                      │  Karacter  │
                      └────────────┘
```

`Kmatrix` and `Karacter` are fully decoupled — a single `Kmatrix` can be shared across multiple `Karacter` instances of matching `dof`, each interpreting the same motion data independently.

---

## 🧬 Data Model

### `Karacter`
Represents a kinematic system such as a robot, articulated rig, or animated character.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `dof_count` | `uint16` | Number of degrees of freedom |
| `accumulator` | `uint32[dof]` | Current accumulated state of every DOF |

### `Kmatrix`
Stores motion as a row-major matrix of signed 32-bit integers.
* **Shape:** `(frames, dof)`
* **Rows:** Represent frames.
* **Columns:** Represent degrees of freedom.
* **Values:** Motion deltas encoded as `int32`.

### `Kimage`
Lossless RGBA representation of a `Kmatrix`.
* **Shape:** `(frames, dof, 4)`
* **Final Dimension:** `[R, G, B, A]` corresponding to the four bytes of an `int32` value.

### `Kframe`
Represents a single motion frame.
* **Shape:** `(dof,)`
* **Implementation:** Direct subclass of `numpy.ndarray` for seamless interoperability with NumPy operations.

---

## 🛠️ Core Functions

All functions live under `kixel.functions` and can be imported in one line:

```python
from kixel.functions import create, clear, flow, update, updatc, state, info, encode, decode
```

| Function | Works on | Description |
| :--- | :--- | :--- |
| `create(kind, *args)` | `Karacter`, `Kframe`, `Kmatrix`, `Kimage` | Instantiates the given type |
| `clear(obj)` | `Karacter`, `Kframe`, `Kmatrix`, `Kimage` | Zeroes out the object's underlying data in-place |
| `flow(index, source)` | `Kmatrix`, `Kimage` | Extracts a single `Kframe` at `index` from a matrix or an image |
| `update(karacter, kframe)` | `Karacter` + `Kframe` | Adds a frame's delta into the karacter's accumulator |
| `updatc(karacter, kframe)` | `Karacter` + `Kframe` | `update()` followed by an automatic `clear()` of the frame |
| `state(karacter)` | `Karacter` | Returns a copy of the current accumulator |
| `info(obj)` | `Karacter`, `Kframe`, `Kmatrix`, `Kimage` | Prints a debug summary of the object |
| `encode(kmatrix)` | `Kmatrix` | Encodes a motion matrix into a lossless `Kimage` |
| `decode(kimage)` | `Kimage` | Decodes a `Kimage` back into the original `Kmatrix` |

Naming convention: frequently used utility verbs are single words (`update`, `clear`, `flow`, `state`, `info`), while rarer construction/conversion operations use `verb_object` naming (`encode`, `decode`).

---

## 🎨 Encoding Strategy

Each motion value is stored as a signed 32-bit integer (`int32`). Kixel converts the value into four bytes using a fixed **big-endian** layout:

```text
       int32 (32 bits)
      ┌────┬────┬────┬────┐
      │ B0 │ B1 │ B2 │ B3 │
      └─┬──┴─┬──┴─┬──┴─┬──┘
        │    │    │    │
        ▼    ▼    ▼    ▼
       [R]  [G]  [B]  [A]   ◀── RGBA Pixel
```

The reverse operation reconstructs the original integer exactly. Because the transformation operates directly on raw bytes:
* ❌ No normalization is performed.
* ❌ No quantization is performed.
* ❌ No rounding occurs.
* ✅ **No precision is lost.**

---

## 🔄 Accumulator Model

Motion frames store **relative changes (deltas)**, while the accumulator stores the **current absolute state**.

For each frame:
```math
accumulator \leftarrow accumulator + delta
```

**Internally:**
* Frame values are represented as `int32`.
* Accumulator values are represented as `uint32`.
* Arithmetic wraps naturally at $2^{32}$.

This provides a compact circular representation suitable for rotational systems.

---

## 🔒 Lossless Guarantee

The following identity **always** holds:

```python
restored = decode(encode(motion))

assert np.array_equal(
    motion.kmatrix,
    restored.kmatrix
) # Returns True
```
*Encoding and decoding preserve every single bit of the original motion data.*

---

## 🚀 Usage Example

```python
import numpy as np
from kixel.classes.karacter import Karacter
from kixel.classes.kmatrix import Kmatrix
from kixel.functions import create, flow, update, encode, decode

# 1. Initialize a kinematic character (e.g., a 6-DOF robot arm)
robot = create(Karacter, 6)

# 2. Generate motion data (independent of any Karacter)
motion = create(Kmatrix, 100, 6)
motion.kmatrix[:] = np.random.randint(-2**31, 2**31 - 1, size=(100, 6), dtype=np.int32)

# 3. Encode to lossless RGBA image
image = encode(motion)

# 4. Decode back to exact original motion data
restored = decode(image)

# 5. Extract a specific frame and update the robot's state
frame = flow(0, motion)
update(robot, frame)
```

---

## 🖼️ PNG Integration

Kixel can persist any `Kimage` as an actual `.png` file on disk, fully losslessly, using `kixel.core.tools.imgtool.png`:

```python
from kixel.core.tools.imgtool.png import save_png, load_png

save_png(image, "motion.png")
loaded = load_png("motion.png")

assert np.array_equal(image.kimage, loaded.kimage)  # Always True
```

PNG's 8-bit RGBA (`color type 6`) channel order is fixed by spec, so byte order is preserved exactly across save/load with no platform-dependent surprises.

---

## ⚙️ Byte Order

Kixel uses a **fixed big-endian representation** for all encoding and decoding operations. This ensures identical results across platforms regardless of native machine endianness.

---

## 🎯 Design Goals

- [x] **Lossless** motion representation
- [x] **Deterministic** encoding and decoding
- [x] **NumPy-first** implementation
- [x] **Explicit** memory layout
- [x] **Platform-independent** byte ordering
- [x] **Compatibility** with image-processing workflows
- [x] **Decoupled** motion data — a `Kmatrix` is not tied to any single `Karacter`

---

## 📊 Status & Future Directions

### Current Status (v0.0.2)
**Implemented features:**
* `Karacter`, `Kmatrix`, `Kimage`, `Kframe` core classes.
* Unified generic functions: `create`, `clear`, `flow`, `update`, `updatc`, `state`, `info`, `encode`, `decode`.
* Lossless image encoding & decoding.
* PNG persistence via `core/tools/imgtool`.
* Accumulator updates with automatic or manual frame clearing.

### Future Directions
Potential areas of exploration include:
* 📦 Dataset generation utilities
* 🧠 Research workflows for vision-based motion understanding
* 📐 Spatiotemporal DOF indexing tools
* 🧩 Additional `core/tools` integrations beyond PNG
