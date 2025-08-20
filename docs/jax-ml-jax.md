<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing with Python

**JAX is a powerful Python library that transforms numerical functions for high-performance computing, making it ideal for machine learning and scientific applications.** ([Original Repository](https://github.com/jax-ml/jax))

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Calculate gradients of native Python and NumPy functions with ease.
*   **Just-In-Time Compilation (JIT):** Compile your Python and NumPy code using XLA for significant performance gains on CPUs, GPUs, and TPUs.
*   **Vectorization (`vmap`):** Efficiently vectorize functions for parallel processing across array axes, simplifying your code and boosting performance.
*   **Scalability:** Scale your computations across multiple devices (TPUs, GPUs) using various parallelization strategies.
*   **Composable Transformations:** Combine `jax.grad`, `jax.jit`, and `jax.vmap` for powerful and flexible computations.
*   **Flexible Control Flow:** Supports differentiation and compilation through loops, branches, recursion, and closures.

## Table of Contents

*   [What is JAX?](#what-is-jax)
*   [Transformations](#transformations)
    *   [Automatic Differentiation with `grad`](#automatic-differentiation-with-grad)
    *   [Compilation with `jit`](#compilation-with-jit)
    *   [Auto-vectorization with `vmap`](#auto-vectorization-with-vmap)
*   [Scaling](#scaling)
*   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
    *   [Supported Platforms](#supported-platforms)
    *   [Installation Instructions](#instructions)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## What is JAX?

JAX is a Python library designed for high-performance numerical computing. It excels in accelerator-oriented array computation and program transformation, making it ideal for large-scale machine learning and scientific computing. It leverages XLA (Accelerated Linear Algebra) for compiling and optimizing your NumPy programs across various hardware accelerators like TPUs and GPUs.

## Transformations

JAX's core strength lies in its ability to transform numerical functions.

### Automatic Differentiation with `grad`

Effortlessly compute reverse-mode gradients using `jax.grad`.

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

You can differentiate to any order with `grad` and use differentiation with Python control flow.  Refer to the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more.

### Compilation with `jit`

Boost performance by compiling your functions using XLA with `jax.jit`.

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

See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more on `jit`.

### Auto-vectorization with `vmap`

Vectorize your functions efficiently with `jax.vmap`.

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices or per-example gradients.

## Scaling

JAX enables scaling your computations across thousands of devices through various parallelization methods, including:
*   **Compiler-based automatic parallelization**
*   **Explicit sharding and automatic partitioning**
*   **Manual per-device programming**

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and
[advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Bits

Be aware of the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform        | CPU            | NVIDIA GPU | Google TPU | AMD GPU | Apple GPU | Intel GPU  |
|-----------------|----------------|------------|------------|---------|-----------|------------|
| Linux x86_64    | yes            | yes        | yes        | yes     | n/a       | experimental |
| Linux aarch64   | yes            | yes        | n/a        | no      | no       | n/a         |
| Mac aarch64     | yes            | n/a        | n/a        | n/a     | experimental | n/a         |
| Windows x86_64  | yes            | no         | n/a        | no      | n/a        | no         |
| Windows WSL2 x86_64 | yes         | experimental | n/a        | no      | n/a       | no         |

### Installation Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for more installation strategies.

## Citing JAX

To cite this repository:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

## Reference Documentation

For in-depth information, consult the [reference documentation](https://docs.jax.dev/).  Explore the [developer documentation](https://docs.jax.dev/en/latest/developer.html) for getting started as a JAX developer.