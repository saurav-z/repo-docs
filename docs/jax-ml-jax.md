<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing with Python

**JAX empowers you to accelerate your numerical computations using Python, offering automatic differentiation, just-in-time compilation, and vectorization for high-performance machine learning and scientific computing.**

[**Original Repository**](https://github.com/jax-ml/jax)

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, including through loops, branches, recursion, and closures. Supports reverse-mode (backpropagation), forward-mode differentiation, and higher-order derivatives.
*   **Just-in-Time Compilation (JIT):** Compile your Python and NumPy code to XLA for execution on TPUs, GPUs, and other accelerators, significantly boosting performance.
*   **Vectorization (vmap):**  Efficiently vectorize functions across array axes, simplifying code and improving performance compared to manual looping.
*   **Scalability:**  Scale computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Composable Transformations:** JAX provides a powerful system for composing function transformations like `grad`, `jit`, and `vmap` for advanced numerical computations.

**Table of Contents**

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

JAX is a high-performance, flexible, and extensible Python library designed for numerical computing. It's built for researchers and practitioners in machine learning, scientific computing, and other fields that require fast, scalable computations.  At its core, JAX is a system for transforming numerical functions.

## Transformations

JAX's core strength lies in its ability to transform functions.

### Automatic Differentiation with `grad`

Use `jax.grad` to efficiently compute gradients:

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

Differentiate to any order with `grad`.

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

You can use differentiation with Python control flow:

```python
def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = jax.grad(abs_val)
print(abs_val_grad(1.0))   # prints 1.0
print(abs_val_grad(-1.0))  # prints -1.0 (abs_val is re-evaluated)
```

See the [JAX Autodiff
Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
and the [reference docs on automatic
differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)
for more.

### Compilation with `jit`

Use XLA to compile your functions end-to-end with
[`jit`](https://docs.jax.dev/en/latest/jax.html#just-in-time-compilation-jit),
used either as an `@jit` decorator or as a higher-order function.

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

Using `jax.jit` constrains the kind of Python control flow
the function can use; see
the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)
for more.

### Auto-vectorization with `vmap`

[`vmap`](https://docs.jax.dev/en/latest/jax.html#vectorization-vmap) maps
a function along array axes.
But instead of just looping over function applications, it pushes the loop down
onto the function’s primitive operations, e.g. turning matrix-vector multiplies into
matrix-matrix multiplies for better performance.

Using `vmap` can save you from having to carry around batch dimensions in your
code:

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

By composing `jax.vmap` with `jax.grad` and `jax.jit`, we can get efficient
Jacobian matrices, or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX provides several ways to scale computations across multiple devices:

*   **Compiler-based automatic parallelization:** The compiler automatically shards data and partitions computation.
*   **Explicit sharding and automatic partitioning:** Define data shardings with JAX types for a global view.
*   **Manual per-device programming:** Have per-device views with explicit collectives.

| Mode | View? | Explicit sharding? | Explicit Collectives? |
|---|---|---|---|
| Auto | Global | ❌ | ❌ |
| Explicit | Global | ✅ | ❌ |
| Manual | Per-device | ✅ | ✅ |

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
[advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Bits

For common issues and limitations, see the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

JAX supports a variety of platforms:

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Instructions

Install JAX using pip:

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

For alternative installation options, including compiling from source or using Docker, consult the [JAX documentation](https://docs.jax.dev/en/latest/installation.html).

## Citing JAX

Cite JAX using the following BibTeX entry:

```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

A paper describing JAX's early development can be found at [SysML
2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

For comprehensive details on the JAX API, refer to the [reference documentation](https://docs.jax.dev/). Developers can find helpful resources in the [developer documentation](https://docs.jax.dev/en/latest/developer.html).