html
<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Machine Learning with Python

JAX is a powerful Python library that enables high-performance numerical computation and machine learning through **composable function transformations and automatic differentiation**. <a href="https://github.com/jax-ml/jax">Explore the JAX repository</a>.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, including support for higher-order derivatives.
*   **Just-In-Time Compilation (JIT):** Compile and optimize your NumPy programs using XLA for significant performance gains on TPUs, GPUs, and other hardware accelerators.
*   **Vectorization:** Simplify your code and improve performance with `vmap` for automatic vectorization across array axes.
*   **Composable Transformations:** Build complex computations by composing `jax.grad`, `jax.jit`, and `jax.vmap`.
*   **Scalability:** Scale your computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, or manual per-device programming.
*   **Flexible Control Flow:** JAX supports differentiation and compilation with Python control flow.

## Core Concepts

*   [Transformations](#transformations)
*   [Scaling](#scaling)

### Transformations

JAX excels at transforming numerical functions, offering core transformations:

*   **`jax.grad` (Automatic Differentiation):**  Efficiently calculates reverse-mode gradients.
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
    Differentiate to any order with `grad`:
    ```python
    print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
    # prints 0.62162673
    ```
*   **`jax.jit` (Compilation):** Compile functions for speed with XLA.
    ```python
    import jax
    import jax.numpy as jnp

    def slow_f(x):
      return x * x + x * 2.0

    x = jnp.ones((5000, 5000))
    fast_f = jax.jit(slow_f)
    %timeit -n10 -r3 fast_f(x)
    %timeit -n10 -r3 slow_f(x)
    ```
*   **`jax.vmap` (Vectorization):** Vectorizes functions for efficient array processing.
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
### Scaling

JAX offers multiple approaches to scale your computations across thousands of devices.
*   **Compiler-based automatic parallelization**
*   **Explicit sharding and automatic partitioning**
*   **Manual per-device programming**
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

## Gotchas and Sharp Edges

Be aware of [common gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

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

See [the documentation](https://docs.jax.dev/en/latest/installation.html)
for installation details, including compiling from source, Docker, conda, and FAQs.

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

*   [API Reference](https://docs.jax.dev/)
*   [Developer Documentation](https://docs.jax.dev/en/latest/developer.html)