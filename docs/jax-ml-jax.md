<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: Accelerate Your Numerical Computing with Powerful Transformations

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**Transformations**](#transformations)
| [**Scaling**](#scaling)
| [**Install guide**](#installation)
| [**Change logs**](https://docs.jax.dev/en/latest/changelog.html)
| [**Reference docs**](https://docs.jax.dev/en/latest/)

JAX is a high-performance Python library that transforms numerical functions, enabling automatic differentiation, just-in-time compilation, and vectorization for accelerated scientific computing and machine learning. **[Learn more at the original repo](https://github.com/jax-ml/jax).**

## Key Features:

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, supporting reverse-mode, forward-mode, and higher-order derivatives.
*   **Just-In-Time Compilation (JIT):** Utilize XLA to compile your NumPy programs, optimizing performance on GPUs, TPUs, and other hardware accelerators.
*   **Vectorization (vmap):** Automatically vectorize functions across array axes, simplifying code and enhancing efficiency.
*   **Composable Transformations:** Combine `jax.grad`, `jax.jit`, and `jax.vmap` for advanced functionalities and optimized workflows.
*   **Scalable Computing:** Scale computations across thousands of devices using compiler-based parallelization, explicit sharding, and manual per-device programming.

## Core Functionality:

JAX offers a suite of powerful transformations to accelerate numerical computations:

### Automatic differentiation with `grad`

Efficiently calculate reverse-mode gradients. Differentiate to any order and use differentiation with Python control flow.

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

See the [JAX Autodiff
Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
and the [reference docs on automatic
differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)
for more.

### Compilation with `jit`

Compile functions end-to-end with `jit` for performance gains.

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

See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)
for more.

### Auto-vectorization with `vmap`

Map a function along array axes to enhance performance.

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

## Scaling

JAX provides multiple strategies for scaling computations:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler chooses how to shard data and partition computation
*   **Explicit sharding and automatic partitioning:** have a global view but data shardings are explicit in JAX types
*   **Manual per-device programming:** have a per-device view of data and computation, and can communicate with explicit collectives.

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

## Gotchas and sharp bits

See the [Gotchas
Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

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

See [the documentation](https://docs.jax.dev/en/latest/installation.html)
for information on alternative installation strategies.

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

## Reference documentation

For details about the JAX API, see the
[reference documentation](https://docs.jax.dev/).

For getting started as a JAX developer, see the
[developer documentation](https://docs.jax.dev/en/latest/developer.html).