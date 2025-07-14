<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing with Python

JAX is a powerful Python library designed for high-performance numerical computing and machine learning, offering automatic differentiation, just-in-time compilation, and vectorization capabilities. **Explore the power of JAX by visiting the [original repository](https://github.com/jax-ml/jax).**

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   **Transformations**
*   **Scaling**
*   **Installation**
*   **Change Logs**
*   **Reference Docs**

## Key Features of JAX

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, including differentiation through loops, branches, recursion, and closures.  Supports reverse-mode (backpropagation) and forward-mode differentiation, composable to any order.
*   **Just-In-Time (JIT) Compilation:** Compile your NumPy programs for accelerated performance on TPUs, GPUs, and other hardware accelerators using XLA.  JIT and automatic differentiation are fully composable.
*   **Vectorization (vmap):**  Automatically vectorize your functions for efficient operations on arrays, optimizing performance by pushing loops down to the primitive operations.
*   **Scalability:** Easily scale computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Extensible System:** At its core, JAX is an extensible system for composable function transformations at scale.

## Transformations: The Core of JAX

JAX offers a suite of powerful function transformations, including `jax.grad`, `jax.jit`, and `jax.vmap`.

### Automatic Differentiation with `grad`

Calculate reverse-mode gradients efficiently.

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

Differentiate to any order and use with Python control flow. See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more.

### Compilation with `jit`

Use XLA for end-to-end function compilation:

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

Map a function along array axes:

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices or per-example gradients.

## Scaling Your Computations

JAX offers multiple modes for scaling computations across devices:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler chooses how to shard data and partition computation.
*   **Explicit sharding and automatic partitioning:** Data shardings are explicit in JAX types.
*   **Manual per-device programming:**  Per-device view of data and computation, with explicit collectives.

| Mode      | View?     | Explicit Sharding? | Explicit Collectives? |
|-----------|-----------|--------------------|-----------------------|
| Auto      | Global    | ❌                 | ❌                     |
| Explicit  | Global    | ✅                 | ❌                     |
| Manual    | Per-device | ✅                 | ✅                     |

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Edges

Refer to the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

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

See the [documentation](https://docs.jax.dev/en/latest/installation.html) for additional installation strategies.

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

Find detailed information about the JAX API in the [reference documentation](https://docs.jax.dev/). For JAX developers, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).