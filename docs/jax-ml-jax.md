<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX empowers you to perform advanced numerical computations, offering automatic differentiation, XLA compilation, and scalable performance for both CPU/GPU/TPU.** [Learn more at the original repository](https://github.com/jax-ml/jax).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

## Key Features:

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions with `jax.grad`, including support for higher-order derivatives and complex control flow.
*   **XLA Compilation:** Compile your Python and NumPy code to optimized machine code using XLA, enabling significant performance gains on various hardware accelerators (TPUs, GPUs, CPUs).
*   **Function Transformation:** Utilize powerful function transformations like `jax.jit` (just-in-time compilation) and `jax.vmap` (vectorization) to optimize and scale your computations.
*   **Scalability:** Scale your computations across thousands of devices with compiler-based automatic parallelization, explicit sharding, and manual per-device programming options.
*   **Hardware Acceleration:**  Seamlessly leverage hardware accelerators like GPUs, TPUs, and other platforms to accelerate numerical computations and machine learning models.

## Core Functionality:

### Transformations

JAX's core strength lies in its ability to transform numerical functions. The main transformations are:

*   **Automatic Differentiation with `grad`**: Efficiently compute reverse-mode gradients. Supports differentiation of Python control flow and can be composed to arbitrary orders.
*   **Compilation with `jit`**: Uses XLA to compile functions for performance gains, often leading to significant speedups, especially for array operations.
*   **Auto-vectorization with `vmap`**: Maps a function along array axes, converting operations into matrix-matrix computations for performance gains.

### Scaling

Scale your workloads across multiple devices with the following techniques:

*   **Compiler-based Automatic Parallelization:** Write code as if using a single global machine; the compiler handles data sharding and computation partitioning.
*   **Explicit Sharding and Automatic Partitioning:** Define data sharding explicitly using JAX types, with a global view.
*   **Manual Per-Device Programming:** Control data and computation at the per-device level, leveraging explicit collectives.

## Installation

### Supported Platforms:

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Instructions:

*   **CPU:** `pip install -U jax`
*   **NVIDIA GPU:** `pip install -U "jax[cuda12]"`
*   **Google TPU:** `pip install -U "jax[tpu]"`
*   **AMD GPU (Linux):** Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).
*   **Mac GPU:** Follow [Apple's instructions](https://developer.apple.com/metal/jax/).
*   **Intel GPU:** Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

## Resources

*   **Documentation:** [Reference Documentation](https://docs.jax.dev/) and [Developer Documentation](https://docs.jax.dev/en/latest/developer.html)
*   **Getting Started:** Explore the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
*   **Common Gotchas:** Check the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)

## Citing JAX

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}