<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning

**JAX is a powerful Python library that transforms numerical functions for high-performance computing, offering automatic differentiation, just-in-time compilation, and vectorization.**  Explore JAX and its capabilities at the [original repository](https://github.com/jax-ml/jax).

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions using `jax.grad`.
*   **Just-In-Time Compilation:** Optimize performance with XLA compilation using `jax.jit` for GPUs, TPUs, and other accelerators.
*   **Vectorization:**  Apply functions efficiently to arrays using `jax.vmap`.
*   **Scalable Computing:** Distribute computations across thousands of devices with compiler-based automatic parallelization, explicit sharding, and manual per-device programming options.

## Key Features

*   **Composable Function Transformations:** Build complex transformations by combining `jax.grad`, `jax.jit`, and `jax.vmap`.
*   **Accelerator-Oriented Array Computation:** Designed for high-performance numerical computing on GPUs, TPUs, and other hardware.
*   **Automatic Differentiation of Any Order:** Calculate derivatives of derivatives without limits.
*   **Control Flow Support:** Differentiate through loops, branches, recursion, and closures.
*   **Flexible Scaling Options:** Choose from automatic parallelization, explicit sharding, or manual per-device programming for optimal performance.

## Transformations

JAX excels at transforming numerical functions. Here are key transformations:

### Automatic Differentiation with `grad`

Use [`jax.grad`](https://docs.jax.dev/en/latest/jax.html#jax.grad) to compute reverse-mode gradients.

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

### Compilation with `jit`

Use XLA to compile your functions end-to-end with
[`jit`](https://docs.jax.dev/en/latest/jax.html#just-in-time-compilation-jit),
used either as an `@jit` decorator or as a higher-order function.

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

### Auto-vectorization with `vmap`

[`vmap`](https://docs.jax.dev/en/latest/jax.html#vectorization-vmap) maps a function along array axes.

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

JAX supports scaling computations across devices through:

*   Compiler-based automatic parallelization
*   Explicit sharding and automatic partitioning
*   Manual per-device programming

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

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform        | CPU         | NVIDIA GPU | Google TPU | AMD GPU (Linux) | Mac GPU | Intel GPU | Windows | Windows WSL2 |
|-----------------|-------------|------------|------------|-----------------|---------|-----------|---------|--------------|
| Support         | Yes         | Yes        | Yes        | Yes             | Yes     | Experimental | Yes     | Experimental |

### Instructions

*   **CPU:** `pip install -U jax`
*   **NVIDIA GPU:** `pip install -U "jax[cuda12]"`
*   **Google TPU:** `pip install -U "jax[tpu]"`
*   **AMD GPU (Linux):** Follow AMD's instructions.
*   **Mac GPU:** Follow Apple's instructions.
*   **Intel GPU:** Follow Intel's instructions.

Detailed instructions are available in the [installation documentation](https://docs.jax.dev/en/latest/installation.html).

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

Access detailed API information in the [reference documentation](https://docs.jax.dev/).