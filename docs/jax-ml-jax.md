<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX empowers you to write and transform numerical computations with ease, unlocking the power of accelerators for high-performance computing and machine learning.**

[![Continuous Integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**Transformations**](#transformations) | [**Scaling**](#scaling) | [**Installation**](#installation) | [**Reference Docs**](https://docs.jax.dev/en/latest/) | [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)

## Key Features

*   **Automatic Differentiation:** Effortlessly calculate gradients of complex Python and NumPy code using `jax.grad`.
*   **Just-In-Time Compilation:** Compile and optimize your functions for speed using `jax.jit` with XLA.
*   **Vectorization:** Efficiently vectorize functions over array axes using `jax.vmap`, enhancing performance.
*   **Accelerator Support:** Leverage GPUs, TPUs, and other hardware accelerators for accelerated computing.
*   **Composable Transformations:** Combine transformations like `grad`, `jit`, and `vmap` to build powerful and efficient computations.
*   **Scalable Computing:** Scale your computations across thousands of devices using compiler-based auto-parallelization, explicit sharding and manual per-device programming.

## What is JAX?

JAX is a Python library designed for high-performance numerical computation and machine learning. It brings together the ease of NumPy with the power of automatic differentiation and just-in-time compilation, all while seamlessly scaling across hardware accelerators like GPUs and TPUs. At its core, JAX provides a flexible system for transforming numerical functions, enabling you to optimize and accelerate your code with minimal effort.

JAX uses [XLA](https://www.openxla.org/xla) to compile and scale your NumPy programs on TPUs, GPUs, and other hardware accelerators. You can compile your own pure functions with [`jax.jit`](#compilation-with-jit). Compilation and automatic differentiation can be composed arbitrarily.

This is a research project, not an official Google product. Expect [sharp edges](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).  Please help by trying it out, [reporting bugs](https://github.com/jax-ml/jax/issues), and letting us know what you think!

```python
import jax
import jax.numpy as jnp

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jax.jit(jax.grad(loss))  # compiled gradient evaluation function
perex_grads = jax.jit(jax.vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads
```

### Contents
*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
*   [Neural Net Libraries](#neural-network-libraries)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## Transformations

JAX provides powerful function transformations to optimize and modify your numerical computations.

### Automatic Differentiation with `grad`

Compute gradients efficiently using `jax.grad`:

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

Differentiation works seamlessly with Python control flow:

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

For more information, see the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Use XLA to compile your functions end-to-end with [`jit`](https://docs.jax.dev/en/latest/jax.html#just-in-time-compilation-jit), used either as an `@jit` decorator or as a higher-order function.

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

[`vmap`](https://docs.jax.dev/en/latest/jax.html#vectorization-vmap) maps a function along array axes. But instead of just looping over function applications, it pushes the loop down onto the function’s primitive operations, e.g. turning matrix-vector multiplies into matrix-matrix multiplies for better performance.

Using `vmap` can save you from having to carry around batch dimensions in your code:

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

JAX offers several methods to scale your computations across thousands of devices:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler handles data sharding and computation partitioning (with user-provided constraints).
*   **Explicit sharding and automatic partitioning:** Use a global view, but define data shardings explicitly with JAX types.
*   **Manual per-device programming:** Have a per-device view of data and computation, allowing communication with explicit collectives.

| Mode      | View?       | Explicit Sharding? | Explicit Collectives? |
| --------- | ----------- | ------------------ | --------------------- |
| Auto      | Global      | ❌                 | ❌                    |
| Explicit  | Global      | ✅                 | ❌                    |
| Manual    | Per-device  | ✅                 | ✅                    |

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

For potential issues and limitations, consult the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

|            | Linux x86\_64 | Linux aarch64 | Mac aarch64  | Windows x86\_64 | Windows WSL2 x86\_64 |
|------------|---------------|---------------|--------------|-----------------|---------------------|
| CPU        | yes           | yes           | yes          | yes             | yes                 |
| NVIDIA GPU | yes           | yes           | n/a          | no              | experimental        |
| Google TPU | yes           | n/a           | n/a          | n/a             | n/a                 |
| AMD GPU    | yes           | no            | n/a          | no              | no                  |
| Apple GPU  | n/a           | no            | experimental | n/a             | n/a                 |
| Intel GPU  | experimental  | n/a           | n/a          | no              | no                  |

### Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for detailed installation instructions, including alternative methods.

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

Note that the version number should match the one in `jax/version.py`.

A SysML 2018 paper describing the early versions of JAX is available [here](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

For comprehensive API details, see the [reference documentation](https://docs.jax.dev/).

For developer-specific information, consult the [developer documentation](https://docs.jax.dev/en/latest/developer.html).

**[Back to Top](#jax-high-performance-numerical-computing-and-machine-learning-in-python)**