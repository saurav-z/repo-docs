<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Machine Learning with Python

[JAX](https://github.com/jax-ml/jax) empowers you to write high-performance numerical computations and machine learning models with ease, offering automatic differentiation, just-in-time compilation, and effortless scaling.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, including derivatives of derivatives, with `jax.grad`.
*   **Just-in-Time Compilation:** Compile your Python and NumPy code for enhanced performance on GPUs, TPUs, and other accelerators using `jax.jit`.
*   **Vectorization:** Apply functions along array axes with `jax.vmap` for efficient operations, eliminating the need for manual batch dimension handling.
*   **Scalable Computation:** Distribute your computations across thousands of devices using compiler-based parallelization, explicit sharding, or manual per-device programming.
*   **Composable Transformations:** Combine automatic differentiation, compilation, and vectorization seamlessly.

## Table of Contents

*   [Transformations](#transformations)
    *   [Automatic differentiation with `grad`](#automatic-differentiation-with-grad)
    *   [Compilation with `jit`](#compilation-with-jit)
    *   [Auto-vectorization with `vmap`](#auto-vectorization-with-vmap)
*   [Scaling](#scaling)
*   [Gotchas and sharp bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
    *   [Supported platforms](#supported-platforms)
    *   [Instructions](#instructions)
*   [Citing JAX](#citing-jax)
*   [Reference documentation](#reference-documentation)

## Transformations

JAX is built around the idea of transforming numerical functions.  Here are three key transformations: `jax.grad`, `jax.jit`, and `jax.vmap`.

### Automatic differentiation with `grad`

Calculate gradients efficiently using `jax.grad`.  You can differentiate to any order and with Python control flow.

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

See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more.

### Compilation with `jit`

Compile functions end-to-end for improved performance using `jax.jit`.

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

See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for important considerations.

### Auto-vectorization with `vmap`

`jax.vmap` automatically maps functions across array axes for efficient batch processing.

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

Composing `jax.vmap` with `jax.grad` and `jax.jit` allows for efficient Jacobian matrices, or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX supports scaling computations across multiple devices through the following modes:

*   **Auto:** Compiler-based automatic parallelization.
*   **Explicit:** Explicit sharding with a global view.
*   **Manual:** Per-device programming with explicit collectives.

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

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and sharp bits

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported platforms

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for additional installation methods.

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

A SysML 2018 paper provides an early description.

## Reference documentation

For detailed API information, see the [reference documentation](https://docs.jax.dev/).  Get started developing with JAX with the [developer documentation](https://docs.jax.dev/en/latest/developer.html).