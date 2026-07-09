‎# 🎬 Kixel
‎
‎> **Lossless image representation for kinematic motion data.**
‎
‎[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
‎[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()
‎[![License](https://img.shields.io/badge/license-MPL--2.0-green.svg)](LICENSE)
‎
‎**Kixel** is a Python library for representing kinematic motion as images without losing numerical precision.
‎
‎The library converts motion data from robots, articulated rigs, or 3D characters into an RGBA image representation that preserves every original bit of information. This allows motion sequences to be processed using image-oriented tools and machine learning pipelines (like ViTs and CNNs) while remaining fully reversible.
‎
‎Unlike traditional approaches that normalize values into a limited image range, Kixel stores each motion value as a full `int32` and encodes its four bytes directly into the RGBA channels of a pixel.
‎
‎---
‎
‎## 📑 Table of Contents
‎- [Motivation](#-motivation)
‎- [Architecture](#-architecture)
‎- [Data Model](#-data-model)
‎- [Encoding Strategy](#-encoding-strategy)
‎- [Kinematic Accumulator Model](#-kinematic-accumulator-model)
‎- [Lossless Guarantee](#-lossless-guarantee)
‎- [Core API](#-core-api)
‎- [Design Goals](#-design-goals)
‎- [License](#-license)
‎
‎---
‎
‎## 💡 Motivation
‎
‎Modern computer vision architectures such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) are optimized for image-like inputs. 
‎
‎Kixel structures kinematic data so that physically adjacent Degrees of Freedom (DOF) occupy adjacent columns in the image (`Kframe`), creating explicit **spatial relationships**. Temporal progression flows down the rows of the image (`Kmatrix`). By mapping motion this way, image-based neural networks can natively extract local and global **spatiotemporal patterns** (such as kinematic acceleration, links, and trajectory fluidities) directly from raw pixels.
‎
‎---
‎
‎## 🏗️ Architecture
‎
‎￼
‎  ┌────────────┐
‎  │  Karacter  │
‎  └─────┬──────┘
‎        │ create()
‎        ▼
‎  ┌────────────┐
‎  │  Kmatrix   │  (frames × dof)
‎  └─────┬──────┘
‎        │ [Encoding Pipeline]
‎        ▼
‎  ┌────────────┐
‎  │   Kimage   │  (frames × dof × 4)
‎  └─────┬──────┘
‎        │ [Decoding Pipeline]
‎        ▼
‎  ┌────────────┐      ┌────────────┐
‎  │  Kmatrix   │─────▶│   Kframe   │ (Slicing / Indexing)
‎  └────────────┘      └─────┬──────┘
‎                            │ clear()
‎                            ▼
‎                      ┌────────────┐
‎                      │  Karacter  │
‎                      └────────────┘
‎
‎￼
‎
‎---
‎
‎## 🧬 Data Model
‎
‎### `Karacter`
‎
‎Represents a kinematic system (e.g., robot or articulated character) defined solely by its configuration.
‎
‎* **`dof_count`** (`uint16`): Number of degrees of freedom.
‎* **`nacc`** (`uint32[dof]`): Accumulator for negative wrap-around boundaries.
‎* **`oacc`** (`uint32[dof]`): Accumulator for positive wrap-around boundaries.
‎* **`sacc`** (`bool[dof]`): Boolean state flag indicator per degree of freedom.
‎
‎### `Kmatrix`
‎
‎Stores motion sequences as a row-major matrix of signed 32-bit integers.
‎
‎* **Shape:** `(frames, dof)`
‎* **Rows:** Represent consecutive frames (Time axis).
‎* **Columns:** Represent degrees of freedom (Spatial axis).
‎* **Values:** Motion deltas encoded as `int32`.
‎
‎### `Kimage`
‎
‎Lossless RGBA representation of a `Kmatrix`.
‎
‎* **Shape:** `(frames, dof, 4)`
‎* **Final Dimension:** `[R, G, B, A]` corresponding to the four bytes of an encoded `int32` motion delta.
‎
‎### `Kframe`
‎
‎Represents a single spatiotemporal motion frame.
‎
‎* **Shape:** `(dof,)`
‎* **Implementation:** Direct subclass of `numpy.ndarray` for seamless, zero-copy interoperability with NumPy operations.
‎
‎---
‎
‎## 🎨 Encoding Strategy
‎
‎Each motion value is stored as a signed 32-bit integer (`int32`). Kixel converts the value into four bytes using a fixed **big-endian** layout:
‎
‎￼
‎       int32 (32 bits)
‎      ┌────┬────┬────┬────┐
‎      │ B0 │ B1 │ B2 │ B3 │
‎      └─┬──┴─┬──┴─┬──┴─┬──┘
‎        │    │    │    │
‎        ▼    ▼    ▼    ▼
‎       [R]  [G]  [B]  [A]   ◀── RGBA Pixel
‎
‎￼
‎
‎The reverse operation reconstructs the original integer exactly. Because the transformation operates directly on raw bytes:
‎
‎* ❌ No normalization is performed.
‎* ❌ No quantization is performed.
‎* ❌ No rounding occurs.
‎* ✅ **No precision is lost.**
‎
‎---
‎
‎## 🔄 Kinematic Accumulator Model
‎
‎Motion frames store **relative changes (deltas)**, while the character state is managed by split accumulators.
‎
‎Arithmetic naturally wraps around at $2^{32}$, providing a compact, cyclic representation well-suited for rotational joint configurations and continuous hardware loop systems.
‎
‎---
‎
‎## 🔒 Lossless Guarantee
‎
‎The following identity **always** holds across the pipeline:
‎
‎￼
‎decoded = decode_image(encode_image(kmatrix))
‎
‎assert np.array_equal(
‎    kmatrix.kmatrix,
‎    decoded.kmatrix
‎) # Returns True
‎
‎￼
‎
‎---
‎
‎## 🛠️ Core API
‎
‎### Factory Function
‎
‎* **`create(kind, *args, kwargs)`**: Universal factory pipeline designed for building and initializing Kixel primitives (`Karacter`, `Kmatrix`, `Kimage`, `Kframe`) dynamically.
‎
‎### State Management
‎
‎* **`clear(obj)`**: Resets internal data buffers. Safely zeros out arrays for `Kframe`, `Kmatrix`, and `Kimage`. When applied to a `Karacter`, it flushes the split mappers (`nacc`, `oacc`) to 0 and resets the state tracker (`sacc`) to `False`.
‎
‎---
‎
‎## ⚙️ Byte Order
‎
‎Kixel enforces a **fixed big-endian representation** across all image encoding and decoding pipelines. This ensures byte-level reproducibility and identical visual texture patterns across different hardware platforms regardless of native host machine endianness.
‎
‎---
‎
‎## 🎯 Design Goals
‎
‎* [x] **Lossless** motion representation (Zero precision loss)
‎* [x] **Deterministic** byte-level encoding and decoding
‎* [x] **Spatiotemporal structure** optimized for ViT/CNN network inputs
‎* [x] **NumPy-first** implementation with direct memory views
‎* [x] **Explicit memory layout** with platform-independent byte ordering
‎* [x] **Graphics-pipeline readiness** (Native compatibility with Blender and image workflows)
