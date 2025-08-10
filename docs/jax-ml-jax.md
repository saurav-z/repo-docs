<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Machine Learning with Python

**JAX is a powerful Python library that enables high-performance numerical computation and program transformation, ideal for machine learning and scientific computing.** ([Original Repository](https://github.com/jax-ml/jax))

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Easily compute gradients of native Python and NumPy functions, supporting reverse-mode (backpropagation) and forward-mode differentiation, even through control flow like loops and conditionals.
*   **Just-In-Time (JIT) Compilation:** Leverage XLA to compile your NumPy programs for acceleration on TPUs, GPUs, and other hardware, optimizing performance.
*   **Vectorization (vmap):** Efficiently vectorize functions, eliminating the need to manage batch dimensions and improving performance with optimized matrix operations.
*   **Scalability:** Supports distributed computing across thousands of devices with compiler-based automatic parallelization, explicit sharding, and manual per-device programming options.

## Core Functionalities & Transformations

JAX transforms numerical functions for high performance. Here's a breakdown:

### Automatic Differentiation with `grad`

Compute gradients efficiently, including higher-order derivatives.

```python
import jax
import jax.numpy as jnp

def tanh(x):
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = jax.grad(tanh)
print(grad_tanh(1.0))
# prints 0.4199743
```

### Compilation with `jit`

Compile functions using XLA for significant speed improvements.

```python
import jax
import jax.numpy as jnp

def slow_f(x):
  # Element-wise ops see a large benefit from fusion
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jax.jit(slow_f)
%timeit -n10 -r3 fast_f(x)
%timeit -n10 -r3 slow_f(x)
```

### Auto-vectorization with `vmap`

Vectorize functions to operate on arrays, not individual elements.

```python
import jax
import jax.numpy as jnp

def l1_distance(x, y):
  assert x.ndim == y.ndim == 1  # only works on 1D inputs
  return jnp.sum(jnp.abs(x - y))

def pairwise_distances(dist1D, xs):
  return jax.vmap(jax.vmap(dist1D, (0, None)), (None, 0))(xs, xs)

xs = jax.random.normal(jax.random.key(0), (100, 3))
dists = pairwise_distances(l1_distance, xs)
dists.shape  # (100, 100)
```

## Scaling Your Computations

JAX provides several methods to scale your computations, including:

*   **Compiler-based automatic parallelization:** Simplifies data and computation sharding with a global view.
*   **Explicit sharding and automatic partitioning:** Allows you to define explicit data shardings.
*   **Manual per-device programming:**  Offers per-device views and explicit control over collectives.

## Gotchas and Sharp Edges

Consult the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for common issues and pitfalls.

## Installation

### Supported Platforms

| Platform        | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|-----------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Installation Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

For alternative installation methods and troubleshooting, see the [official documentation](https://docs.jax.dev/en/latest/installation.html).

## Citing JAX

To cite JAX, please use the following BibTeX entry:

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

*   **Reference Documentation:** [https://docs.jax.dev/](https://docs.jax.dev/)
*   **Developer Documentation:** [https://docs.jax.dev/en/latest/developer.html](https://docs.jax.dev/en/latest/developer.html)