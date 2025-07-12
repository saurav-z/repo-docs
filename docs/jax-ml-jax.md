<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX is a powerful Python library that transforms your NumPy programs for efficient numerical computation, automatic differentiation, and scaling across hardware accelerators like GPUs and TPUs.**  [Explore the JAX Repository](https://github.com/jax-ml/jax)

[![Continuous Integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   [**Transformations**](#transformations)
*   [**Scaling**](#scaling)
*   [**Installation**](#installation)
*   [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features of JAX:

*   **Automatic Differentiation:**  Effortlessly compute gradients of native Python and NumPy functions with `jax.grad`, enabling reverse-mode (backpropagation) and forward-mode differentiation, and higher-order derivatives.
*   **Just-In-Time (JIT) Compilation:** Optimize your code for speed with `jax.jit` using XLA, compiling your functions for efficient execution on accelerators, and compose with automatic differentiation.
*   **Vectorization:**  Apply functions across array axes with `jax.vmap` to boost performance and simplify your code by avoiding explicit loops.
*   **Scalable Computing:**  Scale your computations across multiple devices and accelerators, supporting compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Extensible System:**  Build on composable function transformations, extending the power of your numerical programs.

## Transformations

JAX's core strength lies in its ability to transform numerical functions.  Here are three key transformations:

### Automatic Differentiation with `grad`

Calculate reverse-mode gradients efficiently:

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

And use differentiation with Python control flow.  See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more details.

### Compilation with `jit`

Compile functions using XLA for faster execution:

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

Review the [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more.

### Auto-vectorization with `vmap`

Vectorize your functions along array axes:

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices and per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

Scale computations across thousands of devices using a combination of these approaches:

*   **Compiler-based automatic parallelization:** Program as if using a single machine; the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:**  Explicitly define data shardings with global view using JAX types.
*   **Manual per-device programming:**  Per-device view with explicit collectives.

| Mode          | View?     | Explicit Sharding? | Explicit Collectives? |
|---------------|-----------|--------------------|-----------------------|
| Auto          | Global    | ❌                 | ❌                    |
| Explicit      | Global    | ✅                 | ❌                    |
| Manual        | Per-device| ✅                 | ✅                    |

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

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for common pitfalls.

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

### Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation strategies and answers to frequently asked questions.

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

A SysML 2018 paper provides an early overview: [https://mlsys.org/Conferences/2019/doc/2018/146.pdf](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

Find complete details in the [reference documentation](https://docs.jax.dev/).

For JAX development, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).