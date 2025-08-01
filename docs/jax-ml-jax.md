<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing with Python

JAX is a powerful Python library that brings the speed of accelerators to your numerical computations, offering automatic differentiation, just-in-time compilation, and vectorization for high-performance machine learning and scientific computing.  [Visit the original repository](https://github.com/jax-ml/jax) for more information.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Easily calculate gradients of native Python and NumPy functions with `jax.grad`, enabling complex model training.
*   **Just-In-Time Compilation (JIT):**  Compile your code for optimal performance on GPUs, TPUs, and other hardware using `jax.jit`.
*   **Vectorization with `vmap`:**  Accelerate computations by automatically vectorizing functions across array axes, reducing the need for manual batching.
*   **Scalable Computations:**  Distribute your work across multiple devices and even thousands of accelerators using compiler-based automatic parallelization, explicit sharding, or manual per-device programming.
*   **Composable Transformations:**  Combine automatic differentiation, compilation, and vectorization for advanced numerical computations.
*   **Pythonic Integration:**  Use JAX with native Python and NumPy code, including control flow.
*   **Hardware Acceleration:** Optimize your code to run on various hardware like CPUs, GPUs (NVIDIA, AMD, Apple, Intel), and TPUs.

## Table of Contents

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

## Transformations

JAX excels at transforming numerical functions.

### Automatic differentiation with `grad`

Use `jax.grad` to efficiently compute reverse-mode gradients.

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

You can differentiate to any order with `grad`.

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

You're free to use differentiation with Python control flow.

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

See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more.

### Compilation with `jit`

Use XLA to compile your functions end-to-end with `jit`, used either as an `@jit` decorator or as a higher-order function.

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

Using `jax.jit` constrains the kind of Python control flow the function can use; see the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more.

### Auto-vectorization with `vmap`

`vmap` maps a function along array axes. But instead of just looping over function applications, it pushes the loop down onto the function’s primitive operations, e.g. turning matrix-vector multiplies into matrix-matrix multiplies for better performance.

Using `vmap` can save you from having to carry around batch dimensions in your code.

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

By composing `jax.vmap` with `jax.grad` and `jax.jit`, we can get efficient Jacobian matrices, or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

To scale your computations across thousands of devices, you can use any composition of these:

*   [**Compiler-based automatic parallelization**](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) where you program as if using a single global machine, and the compiler chooses how to shard data and partition computation (with some user-provided constraints);
*   [**Explicit sharding and automatic partitioning**](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html) where you still have a global view but data shardings are explicit in JAX types, inspectable using `jax.typeof`;
*   [**Manual per-device programming**](https://docs.jax.dev/en/latest/notebooks/shard_map.html) where you have a per-device view of data and computation, and can communicate with explicit collectives.

| Mode        | View?      | Explicit sharding? | Explicit Collectives? |
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

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Bits

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform        | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
| --------------- | ------------ | ------------- | ------------ | ---------------- | --------------------- |
| CPU             | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU      | yes          | yes           | n/a          | no             | experimental        |
| Google TPU      | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU         | yes          | no            | n/a          | no             | no                  |
| Apple GPU       | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU       | experimental | n/a           | n/a          | no             | no                  |

### Instructions

| Platform        | Instructions                                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CPU             | `pip install -U jax`                                                                                                                                                     |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                                                                           |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                                                                              |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                                                                             |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                                                                                   |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).                                                           |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for information on alternative installation strategies.

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

A nascent version of JAX, supporting only automatic differentiation and compilation to XLA, was described in a [paper that appeared at SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf). We're currently working on covering JAX's ideas and capabilities in a more comprehensive and up-to-date paper.

## Reference Documentation

For details about the JAX API, see the [reference documentation](https://docs.jax.dev/).

For getting started as a JAX developer, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).