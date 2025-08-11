<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: Accelerate Your Numerical Computing with Python

**JAX empowers you to write high-performance numerical code in Python, with automatic differentiation, XLA compilation, and distributed computing capabilities.** Learn more and contribute on [GitHub](https://github.com/jax-ml/jax).

*   [**Transformations**](#transformations)
*   [**Scaling**](#scaling)
*   [**Installation**](#installation)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)
*   [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)

## What is JAX?

JAX is a powerful Python library designed for high-performance numerical computing and machine learning. It provides a flexible and extensible system for transforming numerical functions, built upon the foundations of NumPy and XLA (Accelerated Linear Algebra). JAX excels in scenarios requiring automatic differentiation, compilation to hardware accelerators like GPUs and TPUs, and distributed computing capabilities.

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions using `jax.grad`. Differentiate through loops, branches, recursion, and closures, with support for arbitrary-order derivatives.
*   **XLA Compilation:** Boost performance by compiling your NumPy programs with XLA using `jax.jit`, enabling execution on TPUs, GPUs, and other hardware accelerators.
*   **Function Transformations:** Compose `jax.grad`, `jax.jit`, and `jax.vmap` (vectorization) to create powerful, optimized computations.
*   **Scaling:** Easily scale your computations across multiple devices and systems with compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Extensible:** JAX's modular design allows for custom transformations and extensions.

## Transformations

JAX's core strength lies in its ability to transform numerical functions.

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

**Further Reading:** [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)

### Compilation with `jit`

Speed up execution using `jax.jit` to compile functions:

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

**Further Reading:** [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)

### Auto-vectorization with `vmap`

Vectorize functions using `vmap`:

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

JAX supports scaling computations across thousands of devices. Choose between:

*   **Auto:** Compiler-based automatic parallelization.
*   **Explicit:** Explicit sharding with a global view.
*   **Manual:** Per-device programming with explicit collectives.

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

**Further Reading:** [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html), [explicit sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html), and [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html).

## Gotchas and Sharp Bits

Be aware of common JAX gotchas.

*   See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform        | Linux x86\_64 | Linux aarch64 | Mac aarch64  | Windows x86\_64 | Windows WSL2 x86\_64 |
| :-------------- | :------------ | :------------ | :----------- | :-------------- | :------------------ |
| CPU             | yes           | yes           | yes          | yes             | yes                 |
| NVIDIA GPU      | yes           | yes           | n/a          | no              | experimental        |
| Google TPU      | yes           | n/a           | n/a          | n/a             | n/a                 |
| AMD GPU         | yes           | no            | n/a          | no              | no                  |
| Apple GPU       | n/a           | no            | experimental | n/a             | n/a                 |
| Intel GPU       | experimental  | n/a           | n/a          | no              | no                  |

### Instructions

| Platform        | Instructions                                                                                                    |
| :-------------- | :-------------------------------------------------------------------------------------------------------------- |
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

**Further Reading:** [Installation documentation](https://docs.jax.dev/en/latest/installation.html)

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

## Reference Documentation

*   [Reference documentation](https://docs.jax.dev/)
*   [Developer documentation](https://docs.jax.dev/en/latest/developer.html)