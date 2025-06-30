<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

JAX is a powerful Python library that transforms and accelerates numerical computation, enabling high-performance scientific computing and cutting-edge machine learning.  [Visit the original repository](https://github.com/jax-ml/jax) for more details.

**Key Features:**

*   **Automatic Differentiation:** Efficiently compute gradients of Python and NumPy functions, supporting reverse-mode (backpropagation), forward-mode, and higher-order derivatives.
*   **Just-In-Time (JIT) Compilation:** Compile your NumPy programs using XLA for optimized execution on CPUs, GPUs, TPUs, and other hardware accelerators.
*   **Vectorization (vmap):**  Automatically vectorize functions for parallel execution, improving performance without modifying your core code.
*   **Scalability:**  Distribute computations across multiple devices and thousands of cores for large-scale numerical tasks.
*   **Composable Transformations:**  Combine `jax.grad`, `jax.jit`, and `jax.vmap` for flexible and powerful program transformations.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Quick Links:**
*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Installation](#installation)
*   [Reference Docs](https://docs.jax.dev/en/latest/)

## Transformations

JAX's core strength lies in its ability to transform numerical functions.

### Automatic differentiation with `grad`

Effortlessly compute reverse-mode gradients using `jax.grad`:

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

You can differentiate to any order and with Python control flow:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0)) #prints 0.62162673

def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = jax.grad(abs_val)
print(abs_val_grad(1.0))   # prints 1.0
print(abs_val_grad(-1.0))  # prints -1.0 (abs_val is re-evaluated)
```

For further details explore the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Leverage XLA to compile functions using `jax.jit`:

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

Consider checking out the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more information.

### Auto-vectorization with `vmap`

Use `vmap` to automatically map a function along array axes:

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

Combine `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

Scale your computations using various methods:
*   **Compiler-based automatic parallelization:**  Program as if using a single machine, and the compiler handles data sharding.
*   **Explicit sharding and automatic partitioning:**  Explicitly define data shardings while still using a global view.
*   **Manual per-device programming:** Have a per-device view of data and computation, with explicit collectives.

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
See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more information.

## Gotchas and Sharp Bits

Check the [Gotchas
Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms
*   CPU: Linux x86_64, Linux aarch64, Mac aarch64, Windows x86_64, Windows WSL2 x86_64
*   NVIDIA GPU: Linux x86_64, Linux aarch64
*   Google TPU: Linux x86_64
*   AMD GPU: Linux x86_64
*   Apple GPU: Mac aarch64
*   Intel GPU: Linux x86_64

### Installation Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

Refer to the [installation documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation strategies, including compiling from source and community-supported conda builds.

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

A foundational paper on JAX, covering automatic differentiation and compilation to XLA, appeared at [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference documentation

For detailed information about the JAX API, see the [reference documentation](https://docs.jax.dev/).  For getting started as a JAX developer, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).