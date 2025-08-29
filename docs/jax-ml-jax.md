<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing with Automatic Differentiation and Compilation

**JAX** is a powerful Python library for accelerated array computation, offering automatic differentiation, just-in-time compilation, and efficient scaling for high-performance numerical computing and machine learning. [<ins>Go to Original Repo</ins>](https://github.com/jax-ml/jax)

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions using `jax.grad`.
*   **Just-in-Time Compilation:** Optimize your code for speed with XLA using `jax.jit`.
*   **Vectorization:** Apply functions across array axes for efficient parallelization using `jax.vmap`.
*   **Scalability:** Scale computations across multiple devices, including TPUs, GPUs, and CPUs, with automatic and explicit parallelization options.
*   **Composable Transformations:** Combine differentiation, compilation, and vectorization for complex, optimized computations.

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
    *   [Instructions](#instructions)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## What is JAX?

JAX is a Python library designed for high-performance numerical computing, particularly well-suited for machine learning and scientific computing. It empowers users with the ability to transform numerical functions with ease, offering functionalities such as automatic differentiation, just-in-time compilation, and auto-vectorization. By leveraging XLA, JAX enables acceleration on various hardware accelerators, including TPUs and GPUs, making it ideal for large-scale computations.

## Transformations

At its core, JAX excels at transforming numerical functions, providing a versatile system for various operations.

### Automatic Differentiation with `grad`

Use `jax.grad` to efficiently compute reverse-mode gradients:

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

JAX allows differentiation to any order and works seamlessly with Python control flow.

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

For more details, see the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Employ XLA to compile functions end-to-end using `jax.jit`, usable as a decorator or a higher-order function.

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

Refer to the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more information.

### Auto-vectorization with `vmap`

`jax.vmap` maps a function along array axes, transforming operations to boost performance.

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

Combining `jax.vmap` with `jax.grad` and `jax.jit` enables the efficient generation of Jacobian matrices and per-example gradients.

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX facilitates scaling computations across diverse devices with these methods:

*   **Compiler-based automatic parallelization:** Program using a single global machine and allow the compiler to handle data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Employ a global view, with data sharding made explicit via JAX types.
*   **Manual per-device programming:** Program with a per-device view of data, allowing for explicit collectives.

| Mode       | View?        | Explicit Sharding? | Explicit Collectives? |
| ---------- | ------------ | ------------------ | --------------------- |
| Auto       | Global       | ❌                | ❌                    |
| Explicit   | Global       | ✅                | ❌                    |
| Manual     | Per-device | ✅                | ✅                    |

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

Consult the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more information.

## Gotchas and Sharp Bits

Review the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for important considerations.

## Installation

### Supported Platforms

|              | Linux x86\_64 | Linux aarch64 | Mac aarch64  | Windows x86\_64 | Windows WSL2 x86\_64 |
|--------------|---------------|---------------|--------------|----------------|---------------------|
| CPU          | yes           | yes           | yes          | yes            | yes                 |
| NVIDIA GPU   | yes           | yes           | n/a          | no             | experimental        |
| Google TPU   | yes           | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU      | yes           | no            | n/a          | no             | no                  |
| Apple GPU    | n/a           | no            | experimental | n/a            | n/a                 |
| Intel GPU    | experimental  | n/a           | n/a          | no             | no                  |

### Instructions

| Platform        | Instructions                                                                                                                               |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                                                       |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                                             |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                                                |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                                                 |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                                                     |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).                             |

Refer to the [documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation methods, including compiling from source, Docker, and Conda.

## Citing JAX

To cite this repository, use:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

The version number should match `jax/version.py`, and the year corresponds to the project's open-source release.

A paper from SysML 2018 introduces an early version of JAX, supporting autodiff and XLA compilation: [link](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

For detailed API information, visit the [reference documentation](https://docs.jax.dev/).  For development guidance, check out the [developer documentation](https://docs.jax.dev/en/latest/developer.html).