<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning

JAX is a powerful Python library that transforms numerical computations for high performance and large-scale machine learning.  Explore its capabilities at the [original JAX repo](https://github.com/jax-ml/jax/).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   **Automatic Differentiation:** Compute gradients of native Python and NumPy functions.
*   **Just-In-Time (JIT) Compilation:** Accelerate your code with XLA compilation on TPUs, GPUs, and more.
*   **Vectorization:** Apply functions across array axes using `vmap` for efficient computations.

## Key Features

JAX excels in several key areas:

*   **Automatic Differentiation:**

    *   Effortlessly calculate gradients (reverse-mode, forward-mode, and higher-order derivatives).
    *   Works with Python control flow (loops, conditionals, recursion).

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

*   **Just-In-Time (JIT) Compilation:**

    *   Compile Python functions for significant performance gains using XLA.
    *   Can be used as a decorator or a higher-order function.

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

*   **Auto-Vectorization (vmap):**

    *   Vectorize functions to operate on batches of data.
    *   Improves performance by applying operations to entire arrays rather than element-wise.

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

*   **Scaling:**

    *   Supports scaling computations across multiple devices (TPUs, GPUs) using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.

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

## Installation

### Supported Platforms

| Platform        | CPU             | NVIDIA GPU | Google TPU | AMD GPU (Linux) | Apple GPU | Intel GPU | Windows CPU | Windows GPU |
|-----------------|-----------------|------------|------------|-----------------|-----------|-----------|-------------|-------------|
| Linux x86_64    | yes             | yes        | yes        | yes             | no        | experimental | yes          | no         |
| Linux aarch64   | yes             | yes        | n/a        | no              | no        | n/a       | yes          | no         |
| Mac aarch64     | yes             | n/a        | n/a        | n/a             | experimental | n/a        | yes          | n/a        |
| Windows x86_64  | yes             | no         | n/a        | no              | n/a        | n/a        | yes          | experimental |
| Windows WSL2    | yes             | experimental| n/a        | no              | n/a        | n/a       | yes          | experimental |

### Installation Instructions

Install JAX using `pip`:

*   **CPU:** `pip install -U jax`
*   **NVIDIA GPU:** `pip install -U "jax[cuda12]"`
*   **Google TPU:** `pip install -U "jax[tpu]"`
*   **AMD GPU (Linux):** Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).
*   **Mac GPU:** Follow [Apple's instructions](https://developer.apple.com/metal/jax/).
*   **Intel GPU:** Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

See the [documentation](https://docs.jax.dev/en/latest/installation.html) for more installation options.

## Additional Resources

*   **Gotchas and Sharp Bits:** [Common Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
*   **Reference Documentation:** [API Documentation](https://docs.jax.dev/)
*   **Developer Documentation:** [Developer Guide](https://docs.jax.dev/en/latest/developer.html)

## Citing JAX

To cite this project:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```