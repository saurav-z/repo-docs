<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX is a powerful Python library that combines automatic differentiation, compilation, and vectorization for accelerated numerical computation, enabling you to scale your machine learning models.**  [Visit the original repository](https://github.com/jax-ml/jax).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:**  Effortlessly compute gradients of complex functions, including those with control flow, loops, and recursion.
*   **Just-In-Time (JIT) Compilation:**  Compile Python and NumPy code to optimize performance on various hardware accelerators like GPUs and TPUs using XLA.
*   **Vectorization (vmap):**  Automatically vectorize functions for efficient parallel processing, reducing the need for manual batching.
*   **Scalable Computations:** Designed to scale your NumPy programs on TPUs, GPUs, and other hardware accelerators.
*   **Composable Transformations:** Build complex computations by composing automatic differentiation, JIT compilation, and vectorization.
*   **Multi-Platform Support:** Supports CPU, NVIDIA GPUs, Google TPUs, AMD GPUs, Apple GPUs, and Intel GPUs.

**Jump to:**
*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Installation](#installation)
*   [Reference Documentation](#reference-documentation)
*   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)

## Transformations

JAX's core strength lies in its ability to transform numerical functions.  Key transformations include:

### Automatic Differentiation with `grad`

Compute reverse-mode gradients efficiently using `jax.grad`:

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

`jax.grad` can be used with Python control flow and to compute higher-order derivatives. See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) for more.

### Compilation with `jit`

Use `jax.jit` for end-to-end compilation with XLA:

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

See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more.

### Auto-vectorization with `vmap`

Use `jax.vmap` to map a function along array axes for improved performance:

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

JAX offers several approaches to scale computations across thousands of devices:

*   **Compiler-based automatic parallelization**:  Program as if using a single global machine, and the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning**:  Employ explicit data sharding with a global view of data.
*   **Manual per-device programming**:  Work with a per-device view of data and computation, utilizing explicit collectives.

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
[advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more details.

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

Refer to [the documentation](https://docs.jax.dev/en/latest/installation.html) for installation details.

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

## Reference documentation

Access the comprehensive [reference documentation](https://docs.jax.dev/) for detailed API information.