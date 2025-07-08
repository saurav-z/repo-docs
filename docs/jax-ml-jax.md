html
<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX** is a powerful Python library for high-performance numerical computing and machine learning, offering automatic differentiation, just-in-time compilation, and automatic vectorization capabilities.  **[Explore the JAX Repository](https://github.com/jax-ml/jax)**.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   **Automatic Differentiation:** Effortlessly calculate gradients of native Python and NumPy functions using `jax.grad`.
*   **Just-In-Time Compilation:** Compile your functions for optimized performance using `jax.jit` with XLA.
*   **Automatic Vectorization:**  Vectorize functions across array axes with `jax.vmap` for efficient parallel computations.
*   **Scalable Computing:** Run your programs on TPUs, GPUs, and other hardware accelerators.
*   **Composable Transformations:**  Combine and chain transformations for advanced numerical computation and machine learning tasks.

## Key Features

*   **Automatic Differentiation (Autograd):**
    *   Effortlessly compute reverse-mode gradients (backpropagation) with `jax.grad`.
    *   Differentiate through loops, branches, recursion, and closures.
    *   Supports arbitrary-order derivatives.
    ```python
    import jax
    import jax.numpy as jnp

    def tanh(x):
      y = jnp.exp(-2.0 * x)
      return (1.0 - y) / (1.0 + y)

    grad_tanh = jax.grad(tanh)
    print(grad_tanh(1.0)) # prints 0.4199743
    ```
    For more details, see the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

*   **Just-In-Time (JIT) Compilation:**
    *   Compile Python functions with XLA using `jax.jit` for significant performance gains.
    *   Uses the `@jit` decorator or can be used as a higher-order function.
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
    For more on control flow with JIT, explore the [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) tutorial.

*   **Automatic Vectorization (vmap):**
    *   Vectorize functions along array axes with `jax.vmap`.
    *   Offers substantial performance benefits by pushing loop operations down to primitive operations.
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
    Compose `jax.vmap` with `jax.grad` and `jax.jit` to get efficient Jacobian matrices or per-example gradients:
    ```python
    per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
    ```

## Scaling Your Computations

JAX supports scaling your computations across numerous devices using:
*   **Compiler-based automatic parallelization** for single-machine programming.
*   **Explicit sharding and automatic partitioning** where data shardings are explicit in JAX types.
*   **Manual per-device programming** with per-device data and computation.

| Mode       | View?      | Explicit Sharding? | Explicit Collectives? |
|------------|------------|--------------------|-----------------------|
| Auto       | Global     | ❌                  | ❌                     |
| Explicit   | Global     | ✅                  | ❌                     |
| Manual     | Per-device | ✅                  | ✅                     |
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
Refer to the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for detailed usage.

## Gotchas and Important Considerations

Be aware of the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) to avoid common pitfalls.

## Installation

### Supported Platforms
Detailed platform support information.

### Installation Instructions

Specific installation instructions based on the platform.

Refer to the [documentation](https://docs.jax.dev/en/latest/installation.html) for further guidance.

## Citing JAX

To cite the JAX project, use the following:
```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

A SysML 2018 paper also offers information,  available at: [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

Find more detailed information in the [reference documentation](https://docs.jax.dev/) and [developer documentation](https://docs.jax.dev/en/latest/developer.html).