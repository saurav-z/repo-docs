html
<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing with Python

**JAX is a powerful Python library that enables accelerated array computation and program transformation for high-performance numerical computing and machine learning.**

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**Transformations**](#transformations)
| [**Scaling**](#scaling)
| [**Installation**](#installation)
| [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
| [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features of JAX

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions.
*   **Just-In-Time Compilation (JIT):** Compile pure functions using XLA for optimized performance on accelerators like GPUs and TPUs.
*   **Auto-Vectorization (vmap):** Efficiently vectorize functions for parallel computation across array axes.
*   **Composable Transformations:** Combine `grad`, `jit`, and `vmap` for powerful and flexible program transformations.
*   **Scalable Computing:**  Leverage automatic parallelization, explicit sharding, and manual per-device programming to scale computations across thousands of devices.

## Core Concepts

JAX excels at transforming numerical functions, with key transformations including:

*   **`jax.grad` (Automatic Differentiation):** Calculates derivatives of any order through Python control flow and complex functions.
    ```python
    import jax
    import jax.numpy as jnp

    def tanh(x):
      y = jnp.exp(-2.0 * x)
      return (1.0 - y) / (1.0 + y)

    grad_tanh = jax.grad(tanh)
    print(grad_tanh(1.0))  # Output: 0.4199743
    ```

*   **`jax.jit` (Compilation):** Compiles functions for significant speedups, especially for element-wise operations.
    ```python
    import jax
    import jax.numpy as jnp

    def slow_f(x):
      return x * x + x * 2.0

    x = jnp.ones((5000, 5000))
    fast_f = jax.jit(slow_f)
    # Example: %timeit -n10 -r3 fast_f(x) will run the compiled function
    # Example: %timeit -n10 -r3 slow_f(x) to compare
    ```

*   **`jax.vmap` (Auto-Vectorization):** Automatically vectorizes functions for efficient parallel execution.
    ```python
    import jax
    import jax.numpy as jnp

    def l1_distance(x, y):
      assert x.ndim == y.ndim == 1
      return jnp.sum(jnp.abs(x - y))

    def pairwise_distances(dist1D, xs):
      return jax.vmap(jax.vmap(dist1D, (0, None)), (None, 0))(xs, xs)

    xs = jax.random.normal(jax.random.key(0), (100, 3))
    dists = pairwise_distances(l1_distance, xs)
    print(dists.shape)  # Output: (100, 100)
    ```

## Scaling Your Computations

JAX provides multiple approaches for scaling:

*   **Compiler-based automatic parallelization:** Lets the compiler handle data sharding and computation partitioning.
*   **Explicit sharding:**  Uses explicit JAX types to define data shardings for better control.
*   **Manual per-device programming:** Provides per-device data and computation views for highly customized parallelization.

## Gotchas and Sharp Edges

Be aware of [common gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) as you develop with JAX.

## Installation

JAX supports various platforms:

### Supported Platforms

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Installation Instructions

*   **CPU:** `pip install -U jax`
*   **NVIDIA GPU:** `pip install -U "jax[cuda12]"`
*   **Google TPU:** `pip install -U "jax[tpu]"`
*   **AMD GPU (Linux):**  Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).
*   **Mac GPU:** Follow [Apple's instructions](https://developer.apple.com/metal/jax/).
*   **Intel GPU:** Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

For detailed information, refer to the [installation documentation](https://docs.jax.dev/en/latest/installation.html).

## Citing JAX

If you use JAX in your research, please cite the project:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

## Further Resources

*   [Reference Documentation](https://docs.jax.dev/): Detailed API reference.
*   [Developer Documentation](https://docs.jax.dev/en/latest/developer.html): Guide for JAX developers.

**Explore the power of JAX and contribute to its development at the [original repository](https://github.com/jax-ml/jax)!**