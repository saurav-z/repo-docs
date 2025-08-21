<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX is a powerful Python library that transforms numerical functions for high-performance computing, offering automatic differentiation, just-in-time compilation, and auto-vectorization capabilities.**  [Learn more about JAX on GitHub](https://github.com/jax-ml/jax).

**Key Features:**

*   **Automatic Differentiation:** Efficiently compute gradients of native Python and NumPy functions using `jax.grad`.  Differentiate through loops, branches, recursion, and closures, to any order.
*   **Just-In-Time Compilation (JIT):**  Compile your NumPy programs for significant performance gains on TPUs, GPUs, and other hardware accelerators with `jax.jit`.
*   **Auto-Vectorization:**  Apply functions across array axes using `jax.vmap` to avoid explicit looping, improving performance and code clarity.
*   **Composable Transformations:** Build complex computations by composing differentiation, compilation, and vectorization.
*   **Scalable Computing:**  Scale your computations across thousands of devices using compiler-based auto-parallelization, explicit sharding, and manual per-device programming.
*   **XLA Integration:** Leverages the XLA compiler for optimized performance on various hardware platforms.

## Core Functionality

JAX provides three core transformation functions: `jax.grad`, `jax.jit`, and `jax.vmap`.

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

Also works with Python control flow.

*   [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
*   [Reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)

### Compilation with `jit`

Compile functions end-to-end using XLA:

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

*   [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)

### Auto-vectorization with `vmap`

Map a function along array axes efficiently:

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

*   Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices or per-example gradients.

## Scaling Your Computations

JAX offers several approaches to scale computations across multiple devices:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler handles data sharding.
*   **Explicit sharding and automatic partitioning:**  Define data sharding with JAX types.
*   **Manual per-device programming:** Implement per-device data and computation with explicit collectives.

| Mode          | View?    | Explicit Sharding? | Explicit Collectives? |
|---------------|----------|--------------------|-----------------------|
| Auto          | Global   | ❌                  | ❌                     |
| Explicit      | Global   | ✅                  | ❌                     |
| Manual        | Per-device | ✅                  | ✅                     |

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

*   [Tutorial](https://docs.jax.dev/en/latest/sharded-computation.html)
*   [Advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html)

## Gotchas and Sharp Edges

*   [Common Gotchas in JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)

## Installation

### Supported Platforms

|                  | Linux x86\_64 | Linux aarch64 | Mac aarch64  | Windows x86\_64 | Windows WSL2 x86\_64 |
|------------------|---------------|---------------|--------------|----------------|---------------------|
| CPU              | yes           | yes           | yes          | yes            | yes                 |
| NVIDIA GPU       | yes           | yes           | n/a          | no             | experimental        |
| Google TPU       | yes           | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU          | yes           | no            | n/a          | no             | no                  |
| Apple GPU        | n/a           | no            | experimental | n/a            | n/a                 |
| Intel GPU        | experimental  | n/a           | n/a          | no             | no                  |

### Instructions

| Platform        | Installation Command                                  |
|-----------------|-------------------------------------------------------|
| CPU             | `pip install -U jax`                                  |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                         |
| Google TPU      | `pip install -U "jax[tpu]"`                            |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md). |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).             |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md). |

*   [Documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation strategies.

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

*   [SysML 2018 Paper](https://mlsys.org/Conferences/2019/doc/2018/146.pdf)

## Reference Documentation

*   [Reference Documentation](https://docs.jax.dev/)
*   [Developer Documentation](https://docs.jax.dev/en/latest/developer.html)