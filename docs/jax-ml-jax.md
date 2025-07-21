<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Program Transformation

**JAX is a powerful Python library for high-performance numerical computing and machine learning, offering automatic differentiation, just-in-time compilation, and automatic vectorization.**  [See the original repo](https://github.com/jax-ml/jax).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   [**Key Features**](#key-features)
*   [**Transformations**](#transformations)
*   [**Scaling**](#scaling)
*   [**Installation**](#installation)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)
*   [**Change logs**](https://docs.jax.dev/en/latest/changelog.html)

## Key Features

*   **Automatic Differentiation:** Easily compute gradients of native Python and NumPy functions, including through loops, branches, and recursion. Supports reverse-mode (backpropagation) and forward-mode differentiation.
*   **Just-in-Time Compilation (JIT):** Compile your NumPy programs to run efficiently on TPUs, GPUs, and other hardware accelerators using XLA.
*   **Auto-vectorization:**  Automatically vectorize functions for efficient parallel processing using `jax.vmap`.
*   **Composable Transformations:**  Build complex computations by combining automatic differentiation, compilation, and vectorization.
*   **Scalable Computing:** Scale your computations across thousands of devices with compiler-based automatic parallelization, explicit sharding, and manual per-device programming options.

## Transformations

JAX's core strength lies in its ability to transform numerical functions.  Key transformations include:

### Automatic Differentiation with `grad`

Compute reverse-mode gradients efficiently:

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

*   Differentiate to any order.
*   Works with Python control flow.

See the [JAX Autodiff
Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
and the [reference docs on automatic
differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)
for more.

### Compilation with `jit`

Compile functions for optimized performance:

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

*   Use `@jit` decorator or higher-order function.
*   Consider control flow limitations when using `jit`.  See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html).

### Auto-vectorization with `vmap`

Apply functions to array axes efficiently:

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

*   Avoid manual batch dimension handling.
*   Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient computations.

## Scaling

JAX provides multiple ways to scale your computations across multiple devices:

*   **Compiler-based automatic parallelization:** Program with a single global view, and the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Use JAX types to define data shardings.
*   **Manual per-device programming:**  Gain a per-device view and use explicit collectives.

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

## Installation

### Supported Platforms

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Installation Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for details on alternative installation methods, including compiling from source, Docker, and more.

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

## Gotchas and Sharp Edges

See the [Gotchas
Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Reference Documentation

For in-depth information, see the [reference documentation](https://docs.jax.dev/).
For developers, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).