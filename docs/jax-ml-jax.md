<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning

**JAX is a powerful Python library that transforms and accelerates numerical computations, enabling large-scale machine learning and scientific computing.**

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**Transformations**](#transformations)
| [**Scaling**](#scaling)
| [**Installation**](#installation)
| [**Change logs**](https://docs.jax.dev/en/latest/changelog.html)
| [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features

*   **Automatic Differentiation:**  Effortlessly calculate gradients of Python and NumPy functions using `jax.grad`.
*   **Just-In-Time Compilation:**  Compile Python functions to accelerate performance with `jax.jit` using XLA.
*   **Vectorization:**  Optimize array operations with `jax.vmap` for efficient computation across multiple data points.
*   **Scalable Computing:**  Distribute computations across multiple devices (GPUs, TPUs, etc.) using compiler-based auto-parallelization and explicit sharding.
*   **Composable Transformations:**  Combine transformations (e.g., `grad`, `jit`, `vmap`) for advanced operations and optimization.
*   **Accelerator Support:**  Leverages hardware accelerators like GPUs and TPUs for significant speedups.

## Transformations

JAX excels at transforming numerical functions through several powerful mechanisms.

### Automatic Differentiation with `grad`

Effortlessly calculate reverse-mode gradients using `jax.grad`:

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

Differentiate to any order:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

Use differentiation with Python control flow:

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

Compile functions end-to-end with
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

JAX supports scaling computations across multiple devices with various methods:

*   **Compiler-based automatic parallelization:**  Program as if using a single machine, and the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Define data shardings explicitly in JAX types.
*   **Manual per-device programming:**  Control data and computation on each device with explicit collectives.

| Mode        | View?      | Explicit sharding? | Explicit Collectives? |
|-------------|------------|--------------------|------------------------|
| Auto        | Global     | ❌                 | ❌                     |
| Explicit    | Global     | ✅                 | ❌                     |
| Manual      | Per-device | ✅                 | ✅                     |

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

Be aware of potential limitations and quirks: [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform       | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|----------------|--------------|---------------|--------------|----------------|---------------------|
| CPU            | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU     | yes          | yes           | n/a          | no             | experimental        |
| Google TPU     | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU        | yes          | no            | n/a          | no             | no                  |
| Apple GPU      | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU      | experimental | n/a           | n/a          | no             | no                  |

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
for alternative installation strategies. These include compiling
from source, installing with Docker, using other versions of CUDA, a
community-supported conda build, and answers to some frequently-asked questions.

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

In the above bibtex entry, names are in alphabetical order, the version number
is intended to be that from [jax/version.py](../main/jax/version.py), and
the year corresponds to the project's open-source release.

A nascent version of JAX, supporting only automatic differentiation and
compilation to XLA, was described in a [paper that appeared at SysML
2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf). We're currently working on
covering JAX's ideas and capabilities in a more comprehensive and up-to-date
paper.

## Reference Documentation

For complete API details, consult the [reference documentation](https://docs.jax.dev/).

For developers, explore the [developer documentation](https://docs.jax.dev/en/latest/developer.html).

[Back to top](https://github.com/jax-ml/jax)
```
Key improvements and optimizations:

*   **SEO Optimization:** Keywords like "JAX," "numerical computing," "machine learning," "automatic differentiation," "GPU," "TPU," "compilation," and "vectorization" are strategically placed in headings, subheadings, and body text.
*   **Concise Summary:**  The one-sentence hook is at the beginning to grab the reader's attention.
*   **Clear Headings and Structure:**  The use of headings, subheadings, and bullet points makes the README easy to scan and understand.
*   **Focus on Benefits:** Key features are presented in a way that emphasizes their value (e.g., "Effortlessly calculate gradients...").
*   **Code Examples:** Kept code examples to showcase functionality
*   **Call to Action / Back to Top:** Added a link back to the original repo at the end.
*   **Installation Highlights**: Added a table for easy platform and installation selection.
*   **Improved Language**: Reworded for clarity and conciseness.
*   **Updated Information:** Incorporated information from the original README.