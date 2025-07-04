<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: Accelerate Your Numerical Computing with Powerful Transformations

[JAX](https://github.com/jax-ml/jax) is a high-performance Python library that transforms numerical computations, perfect for machine learning and scientific computing.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   [**Transformations Overview**](#transformations)
*   [**Scaling Your Computations**](#scaling)
*   [**Installation Guide**](#installation)
*   [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features of JAX

*   **Automatic Differentiation:** Easily compute gradients of native Python and NumPy functions with `jax.grad`, supporting arbitrary order differentiation and control flow.
*   **Just-In-Time (JIT) Compilation:** Compile your Python functions with XLA using `jax.jit` to accelerate performance on GPUs, TPUs, and other accelerators.
*   **Vectorization:** Apply functions to array axes with `jax.vmap` for efficient, vectorized operations without manual looping.
*   **Scalable Computing:** Scale your computations across multiple devices using compiler-based parallelization, explicit sharding, and manual per-device programming.

## Transformations

At its core, JAX is built on the principle of transforming numerical functions.  Key transformations include:

### Automatic Differentiation with `grad`

Effortlessly compute reverse-mode gradients using `jax.grad`.

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

You can chain `grad` for higher-order derivatives:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

Differentiate through Python control flow:

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

Learn more in the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [automatic differentiation documentation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Use XLA to compile functions for significant performance gains using  `jax.jit`:

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

For further information, check out the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)

### Auto-vectorization with `vmap`

Use `vmap` to efficiently map functions across array axes:

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

Compose `jax.vmap`, `jax.grad`, and `jax.jit` for efficient per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX offers various methods to scale computations across thousands of devices:

*   **Compiler-based automatic parallelization:**  Write code as if on a single machine, and the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Use JAX types to explicitly define data shardings.
*   **Manual per-device programming:**  Develop per-device views of data and computation, using explicit collectives for communication.

| Mode       | View?      | Explicit sharding? | Explicit Collectives? |
|------------|------------|--------------------|-----------------------|
| Auto       | Global     | ❌                  | ❌                     |
| Explicit   | Global     | ✅                  | ❌                     |
| Manual     | Per-device | ✅                  | ✅                     |

Example:

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
[advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more details.

## Gotchas and Sharp Bits

See the [Gotchas
Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) to be aware of limitations.

## Installation

### Supported Platforms

JAX supports a wide range of platforms:

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

### Installation Instructions

Install JAX using pip:

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

Refer to the [documentation](https://docs.jax.dev/en/latest/installation.html) for detailed instructions, including compiling from source, using Docker, conda builds, and answers to FAQs.

## Citing JAX

To cite this repository, use the following:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

A SysML 2018 paper describes an early version of JAX: [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

Explore the comprehensive [reference documentation](https://docs.jax.dev/) for in-depth information on the JAX API.  Find resources to help you get started in the [developer documentation](https://docs.jax.dev/en/latest/developer.html).