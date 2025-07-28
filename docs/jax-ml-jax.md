<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Program Transformation in Python

JAX is a powerful Python library that transforms and accelerates numerical computations, ideal for large-scale machine learning and scientific computing.  Explore the original repository: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions using `jax.grad`, supporting arbitrary orders and control flow.
*   **Just-In-Time Compilation (JIT):** Speed up your code with XLA compilation using `jax.jit`, optimizing performance across various hardware accelerators like GPUs and TPUs.
*   **Vectorization (`vmap`):** Efficiently vectorize functions across array axes, simplifying code and improving performance, by automatically pushing loops down to primitive operations.
*   **Scalable Computing:** Leverage automatic parallelization, explicit sharding, and per-device programming for large-scale computations across multiple devices.
*   **Extensible Transformations:** Build custom function transformations by composing existing ones.

## Table of Contents

*   [Transformations](#transformations)
    *   [Automatic Differentiation with `grad`](#automatic-differentiation-with-grad)
    *   [Compilation with `jit`](#compilation-with-jit)
    *   [Auto-vectorization with `vmap`](#auto-vectorization-with-vmap)
*   [Scaling](#scaling)
*   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
    *   [Supported Platforms](#supported-platforms)
    *   [Instructions](#instructions)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## Transformations

JAX is built around the concept of composable function transformations.  This section covers the core transformation tools:

### Automatic Differentiation with `grad`

Calculate gradients with ease using `jax.grad`.

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

Differentiate to any order and integrate with Python control flow.

### Compilation with `jit`

Compile your functions for significant performance gains using XLA and `jax.jit`.

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

For more, see the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html).

### Auto-vectorization with `vmap`

Vectorize functions with `jax.vmap` to apply them across array axes.

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices and per-example gradients.

## Scaling

JAX facilitates scaling your computations across thousands of devices using:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine.
*   **Explicit sharding and automatic partitioning:** Define data shardings with JAX types.
*   **Manual per-device programming:** Achieve a per-device view of data and computation.

| Mode        | View?      | Explicit Sharding? | Explicit Collectives? |
| ----------- | ---------- | ------------------ | --------------------- |
| Auto        | Global     | ❌                 | ❌                    |
| Explicit    | Global     | ✅                 | ❌                    |
| Manual      | Per-device | ✅                 | ✅                    |

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

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and
[advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for further details.

## Gotchas and Sharp Bits

Familiarize yourself with the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) to avoid common pitfalls.

## Installation

### Supported Platforms

JAX offers support for a variety of platforms, including:

| Platform        | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
| --------------- | ------------ | ------------- | ------------ | -------------- | ------------------- |
| CPU             | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU      | yes          | yes           | n/a          | no             | experimental        |
| Google TPU      | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU         | yes          | no            | n/a          | no             | no                  |
| Apple GPU       | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU       | experimental | n/a           | n/a          | no             | no                  |

### Instructions

Follow these instructions for installation:

| Platform        | Instructions                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------- |
| CPU             | `pip install -U jax`                                                                                          |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                   |
| AMD GPU (Linux) | Refer to [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                  |
| Mac GPU         | Consult [Apple's instructions](https://developer.apple.com/metal/jax/).                                        |
| Intel GPU       | See [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md). |

Explore [the documentation](https://docs.jax.dev/en/latest/installation.html) for comprehensive installation guidance.

## Citing JAX

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

A SysML 2018 paper provided an early description of JAX ([https://mlsys.org/Conferences/2019/doc/2018/146.pdf](https://mlsys.org/Conferences/2019/doc/2018/146.pdf)).

## Reference Documentation

Access comprehensive API details in the [reference documentation](https://docs.jax.dev/).

For development insights, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).