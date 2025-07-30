html
<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

JAX is a powerful Python library revolutionizing numerical computing and machine learning with its ability to transform and accelerate array computations.  [Explore the JAX Repository](https://github.com/jax-ml/jax)

[![Continuous Integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions with `jax.grad`.
*   **Just-In-Time Compilation:** Accelerate code with `jax.jit` using XLA for optimal performance on TPUs, GPUs, and other accelerators.
*   **Vectorization:**  Efficiently vectorize functions using `jax.vmap` for parallel processing across array axes.
*   **Scalability:**  Scale computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Composable Transformations:** Build complex computations with arbitrarily composed transformations like `jax.grad`, `jax.jit`, and `jax.vmap`.

## Table of Contents

*   [What is JAX?](#what-is-jax)
    *   [Transformations](#transformations)
        *   [Automatic differentiation with `grad`](#automatic-differentiation-with-grad)
        *   [Compilation with `jit`](#compilation-with-jit)
        *   [Auto-vectorization with `vmap`](#auto-vectorization-with-vmap)
    *   [Scaling](#scaling)
    *   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
    *   [Supported Platforms](#supported-platforms)
    *   [Instructions](#instructions)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## What is JAX?

JAX is a Python library designed for high-performance numerical computing and machine learning.  It provides powerful program transformations and is built on the foundation of accelerator-oriented array computation. JAX offers automatic differentiation, just-in-time compilation, and auto-vectorization to supercharge your numerical programs.  It's designed for both reverse-mode differentiation (backpropagation) and forward-mode differentiation, with the ability to compose them to any order.  JAX compiles your NumPy programs on TPUs, GPUs, and other hardware accelerators using XLA, boosting performance and scalability.

### Transformations

JAX's core strength lies in its ability to transform numerical functions. The library provides several powerful transformations, including `jax.grad`, `jax.jit`, and `jax.vmap`.

#### Automatic differentiation with `grad`

Use `jax.grad` for efficient reverse-mode gradients.

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

You can differentiate to any order with `grad`:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

#### Compilation with `jit`

Compile your functions end-to-end with `jit`.

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

#### Auto-vectorization with `vmap`

Use `vmap` to map a function along array axes for improved performance.

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

### Scaling

Scale computations across thousands of devices using a combination of approaches:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler manages data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Define data shardings explicitly using JAX types.
*   **Manual per-device programming:** Have a per-device view of data and computation with explicit collectives.

| Mode        | View?    | Explicit sharding? | Explicit Collectives? |
|-------------|----------|--------------------|-----------------------|
| Auto        | Global   | ❌                  | ❌                     |
| Explicit    | Global   | ✅                  | ❌                     |
| Manual      | Per-device | ✅                  | ✅                     |

```python
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
mesh = jax.make_mesh((8,), ('data',), axis_types=(AxisType.Explicit,))
set_mesh(mesh)

# parameters are sharded for FSDP:
for W, b in params:
  print(f'{jax.typeof(W)}')  # f32[512@data,512]
  print(f'{jax.typeof(b)}')  # f32[512]

# shard data for batch parallelism:
inputs, targets = jax.device_put((inputs, targets), P('data'))

# evaluate gradients, automatically parallelized!
gradfun = jax.jit(jax.grad(loss))
param_grads = gradfun(params, (inputs, targets))
```

### Gotchas and Sharp Bits

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform        | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|-----------------|--------------|---------------|--------------|----------------|---------------------|
| CPU             | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU      | yes          | yes           | n/a          | no             | experimental        |
| Google TPU      | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU         | yes          | no            | n/a          | no             | no                  |
| Apple GPU       | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU       | experimental | n/a           | n/a          | no             | no                  |

### Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html)
for information on alternative installation strategies.

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

For detailed information about the JAX API, consult the [reference documentation](https://docs.jax.dev/).
For those starting out as JAX developers, consult the [developer documentation](https://docs.jax.dev/en/latest/developer.html).