<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing with Python

**JAX** is a powerful Python library designed for high-performance numerical computing and machine learning, enabling automatic differentiation and array computation on accelerators like GPUs and TPUs. Explore the capabilities of JAX and accelerate your numerical computations! ([Original Repo](https://github.com/jax-ml/jax))

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Easily compute gradients of complex Python and NumPy functions, including those with loops, branches, recursion, and closures.
*   **Just-In-Time (JIT) Compilation:** Compile your Python and NumPy code using XLA for significant performance gains on various hardware accelerators (GPUs, TPUs, etc.).
*   **Vectorization (vmap):** Efficiently vectorize functions to process data in parallel, simplifying your code and improving performance.
*   **Scalability:** Scale your computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, or manual per-device programming.
*   **Extensible Transformations:** Compose transformations like `jax.grad`, `jax.jit`, and `jax.vmap` for flexible and powerful numerical computations.

## Core Concepts

JAX leverages key transformations to enable flexible and efficient numerical computation.

### Automatic Differentiation with `grad`

Compute gradients of your functions with ease:

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

You can differentiate to any order:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) for more.

### Compilation with `jit`

Optimize performance by compiling functions using XLA:

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

Apply functions efficiently to batches of data:

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

## Scaling

JAX offers various methods for scaling your computations:

*   **Compiler-based automatic parallelization:** Program with a global view, and the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Use JAX types to explicitly define data shardings.
*   **Manual per-device programming:** Control data and computation on a per-device level with explicit collectives.

## Installation

### Supported Platforms

| Platform          | CPU          | NVIDIA GPU | Google TPU | AMD GPU (Linux) | Apple GPU (Mac) | Intel GPU     | Windows CPU | Windows GPU (NVIDIA) | Windows GPU (AMD) |
| ----------------- | ------------ | ---------- | ---------- | --------------- | --------------- | ------------- | ----------- | -------------------- | ----------------- |
| Linux x86_64      | yes          | yes        | yes        | yes             | n/a             | experimental  | no          | no                   | no                |
| Linux aarch64     | yes          | yes        | n/a        | no              | no              | n/a           | no          | no                   | no                |
| Mac aarch64       | yes          | n/a        | n/a        | n/a             | experimental    | n/a           | no          | no                   | no                |
| Windows x86_64    | yes          | no         | n/a        | no              | n/a             | no            | yes         | experimental         | no                |
| Windows WSL2 x86_64 | yes          | yes        | n/a        | no              | n/a             | no            | yes         | experimental         | no                |

### Installation Instructions

*   **CPU:** `pip install -U jax`
*   **NVIDIA GPU:** `pip install -U "jax[cuda12]"` (replace cuda12 with your cuda version)
*   **Google TPU:** `pip install -U "jax[tpu]"`
*   **AMD GPU (Linux):** Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).
*   **Mac GPU:** Follow [Apple's instructions](https://developer.apple.com/metal/jax/).
*   **Intel GPU:** Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for more installation options.

## Additional Resources

*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Gotchas and Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
*   [Neural Network Libraries](#neural-network-libraries)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](https://docs.jax.dev/)

## Citing JAX

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}