<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

[JAX](https://github.com/jax-ml/jax) is a powerful Python library revolutionizing numerical computing with its ability to transform and accelerate your code. Designed for high-performance computing and large-scale machine learning, JAX allows you to write efficient and scalable code with ease.

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, supporting arbitrary order derivatives.
*   **Just-In-Time Compilation (JIT):** Compile and optimize your NumPy programs using XLA for blazing-fast performance on TPUs, GPUs, and other accelerators.
*   **Auto-Vectorization (vmap):** Vectorize your functions automatically, enabling efficient parallel execution without manual batching.
*   **Scalable Parallelization:** Easily scale computations across thousands of devices through compiler-based auto-parallelization, explicit sharding, or manual per-device programming.
*   **Composable Transformations:** Combine automatic differentiation, JIT compilation, and vectorization for maximum flexibility and performance.

**[Transformations](#transformations) | [Scaling](#scaling) | [Installation](#installation) | [Reference Docs](https://docs.jax.dev/en/latest/)**

## Core Concepts:

*   **Transformations:** JAX's core strength lies in its ability to transform numerical functions.  `jax.grad`, `jax.jit`, and `jax.vmap` are the primary tools.

    *   **Automatic Differentiation with `grad`:**  Efficiently compute gradients.  Differentiate through control flow, recursion, and more.

        ```python
        import jax
        import jax.numpy as jnp

        def tanh(x):
          y = jnp.exp(-2.0 * x)
          return (1.0 - y) / (1.0 + y)

        grad_tanh = jax.grad(tanh)
        print(grad_tanh(1.0))  # Output: 0.4199743
        ```

    *   **Compilation with `jit`:**  Speed up function execution with just-in-time (JIT) compilation using XLA.

        ```python
        import jax
        import jax.numpy as jnp

        def slow_f(x):
          return x * x + x * 2.0

        x = jnp.ones((5000, 5000))
        fast_f = jax.jit(slow_f)
        %timeit -n10 -r3 fast_f(x)  # Faster
        %timeit -n10 -r3 slow_f(x)  # Slower
        ```

    *   **Auto-vectorization with `vmap`:** Automatically vectorize functions for efficient operations on batches of data.

        ```python
        import jax
        import jax.numpy as jnp

        def l1_distance(x, y):
          return jnp.sum(jnp.abs(x - y))

        def pairwise_distances(dist1D, xs):
          return jax.vmap(jax.vmap(dist1D, (0, None)), (None, 0))(xs, xs)

        xs = jax.random.normal(jax.random.key(0), (100, 3))
        dists = pairwise_distances(l1_distance, xs)
        dists.shape  # Output: (100, 100)
        ```

## Scaling

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler chooses how to shard data and partition computation.
*   **Explicit sharding and automatic partitioning:** Data shardings are explicit in JAX types, inspectable using `jax.typeof`.
*   **Manual per-device programming:** Per-device view of data and computation.

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

Be aware of potential [gotchas and sharp edges](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) as you use JAX.

## Installation

### Supported Platforms

| Platform        | CPU           | NVIDIA GPU | Google TPU | AMD GPU (Linux) | Mac GPU (Experimental) | Intel GPU (Experimental) |
|-----------------|---------------|------------|------------|-----------------|------------------------|--------------------------|
| Linux x86_64    | yes           | yes        | yes        | yes             | no                     | experimental             |
| Linux aarch64   | yes           | yes        | n/a        | no              | no                     | n/a                      |
| Mac aarch64     | yes           | n/a        | n/a        | n/a             | experimental           | n/a                      |
| Windows x86_64  | yes           | no         | n/a        | no              | n/a                    | no                       |
| Windows WSL2 x86_64 | yes      | experimental| n/a         | no              | n/a                    | no                       |

### Installation Instructions

*   **CPU:** `pip install -U jax`
*   **NVIDIA GPU:** `pip install -U "jax[cuda12]"`
*   **Google TPU:** `pip install -U "jax[tpu]"`
*   **AMD GPU (Linux):** Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).
*   **Mac GPU:** Follow [Apple's instructions](https://developer.apple.com/metal/jax/).
*   **Intel GPU:** Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

Refer to the [installation documentation](https://docs.jax.dev/en/latest/installation.html) for further details.

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

Explore the comprehensive [reference documentation](https://docs.jax.dev/) for detailed API information.

For contributions, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).