<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing with Automatic Differentiation

**JAX is a powerful Python library for high-performance numerical computing, offering automatic differentiation, just-in-time compilation, and easy scaling for machine learning and scientific computing.** Explore JAX's capabilities and how it can accelerate your projects.

[![Continuous Integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**Transformations**](#transformations)
| [**Scaling**](#scaling)
| [**Installation**](#installation)
| [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
| [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features of JAX

*   **Automatic Differentiation:** Effortlessly compute gradients of complex functions with `jax.grad`.
*   **Just-in-Time Compilation (JIT):** Compile Python and NumPy code for optimized performance on accelerators (TPUs, GPUs).
*   **Vectorization with `vmap`:** Automatically vectorize functions for efficient processing of batches of data.
*   **Scalable Computing:** Distribute computations across multiple devices and accelerators.
*   **Composable Transformations:** Combine automatic differentiation, JIT compilation, and vectorization for powerful workflows.
*   **Flexible Control Flow:** Differentiate through loops, branches, recursion, and closures.

## Why Use JAX?

JAX provides a flexible and efficient platform for numerical computation, particularly well-suited for:

*   **Machine Learning:** Rapidly prototype and train models with automatic differentiation and efficient hardware utilization.
*   **Scientific Computing:** Accelerate computationally intensive simulations and analyses.
*   **Research:** Explore new algorithms and models with ease and speed.

## Getting Started

JAX is built upon the foundations of NumPy, extended with features like automatic differentiation. This example shows how to easily calculate the derivative of a function:

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

### Contents

*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
*   [Neural Network Libraries](#neural-network-libraries)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## Transformations

JAX's core strength lies in its ability to transform numerical functions using a suite of tools. The most fundamental include: `jax.grad` for automatic differentiation, `jax.jit` for just-in-time compilation, and `jax.vmap` for vectorization.

### Automatic Differentiation with `grad`

Use `jax.grad` to compute reverse-mode gradients for efficient optimization and analysis. It supports differentiation of arbitrary code, including Python control flow.

```python
import jax
import jax.numpy as jnp

def tanh(x):
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = jax.grad(tanh)
print(grad_tanh(1.0))  # prints 0.4199743
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0)) # prints 0.62162673
```

For further exploration, see the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [automatic differentiation reference documentation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Leverage XLA to compile your functions for enhanced performance using the `@jit` decorator or as a higher-order function. This enables significant speedups, especially for operations that benefit from fusion.

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

For details, refer to the [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) tutorial.

### Auto-vectorization with `vmap`

Use `vmap` to automatically vectorize functions across array axes, efficiently handling batches without explicit looping.

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

`vmap` also can be combined with `jax.grad` and `jax.jit` for effective Jacobian matrices or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX facilitates scalable computations across thousands of devices via multiple methods:
*   **Compiler-based automatic parallelization:** Write code as if using a single machine, and the compiler handles sharding and computation partitioning, with user-provided constraints.
*   **Explicit sharding and automatic partitioning:** Maintain a global view of data with explicit shardings defined within JAX types, inspectable using `jax.typeof`.
*   **Manual per-device programming:** Work with a per-device perspective of data and computation, capable of communication via explicit collectives.

| Mode      | View?    | Explicit Sharding? | Explicit Collectives? |
|-----------|----------|--------------------|-----------------------|
| Auto      | Global   | ❌                  | ❌                     |
| Explicit  | Global   | ✅                  | ❌                     |
| Manual    | Per-device| ✅                  | ✅                     |

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

Explore the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for further insights.

## Gotchas and Sharp Bits

Be aware of potential challenges by consulting the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

JAX supports various platforms, providing CPU, GPU, and TPU support.

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Installation Instructions

Choose the appropriate command for your hardware:

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See the [installation documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation methods, including compiling from source, and instructions for different CUDA versions and community-supported builds.

## Neural Network Libraries
JAX is compatible with various neural network libraries:
*   Flax
*   Haiku

## Citing JAX

If you use JAX in your research, please cite the following:

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

For detailed information on the JAX API, explore the [reference documentation](https://docs.jax.dev/).

For developers, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).

**[Visit the JAX GitHub Repository](https://github.com/jax-ml/jax) for more information and to contribute.**