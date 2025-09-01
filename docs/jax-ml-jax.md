<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing with Automatic Differentiation

JAX is a Python library for accelerating numerical computation, offering automatic differentiation and XLA compilation for high-performance machine learning and scientific computing.  Learn more and contribute at the [original repo](https://github.com/jax-ml/jax).

*   **Automatic Differentiation:** Compute gradients of Python and NumPy code, including through loops, branches, and recursion.
*   **XLA Compilation:** Compile and optimize NumPy programs for TPUs, GPUs, and other accelerators using XLA.
*   **Function Transformations:**  Compose `jax.grad`, `jax.jit`, and `jax.vmap` to transform your code efficiently.
*   **Scalable Computing:** Scale computations across thousands of devices with compiler-based automatic parallelization, explicit sharding, and manual per-device programming options.
*   **Hardware Support:**  Run JAX on CPUs, NVIDIA GPUs, Google TPUs, AMD GPUs, Apple GPUs, and Intel GPUs (with varying levels of support).

## Key Features

### Automatic Differentiation with `grad`

Effortlessly compute gradients of arbitrary Python and NumPy functions:

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

Differentiate to any order and use differentiation with Python control flow:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

See the [JAX Autodiff
Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
and the [reference docs on automatic
differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)
for more.

### Compilation with `jit`

Compile your functions end-to-end with `jit` to boost performance, and use it as an `@jit` decorator or a higher-order function.

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

See the [tutorial on Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more.

### Auto-vectorization with `vmap`

Vectorize functions along array axes with `vmap`, improving performance by pushing the loop down onto the function’s primitive operations:

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX provides flexibility to scale your computations across numerous devices. Choose from several modes:

*   **Auto:** Compiler-based automatic parallelization.
*   **Explicit:** Explicit sharding and automatic partitioning.
*   **Manual:** Per-device programming.

| Mode        | View?      | Explicit sharding? | Explicit Collectives? |
|-------------|------------|--------------------|------------------------|
| Auto        | Global     | ❌                 | ❌                     |
| Explicit    | Global     | ✅                 | ❌                     |
| Manual      | Per-device | ✅                 | ✅                     |

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

For further details see the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and the [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html).

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

For additional information see the [installation documentation](https://docs.jax.dev/en/latest/installation.html), which includes options for alternative installation strategies, such as compiling from source and installing with Docker.

## Gotchas and Sharp Edges

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

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

A foundational paper on JAX, covering only automatic differentiation and compilation to XLA, can be found at [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

*   [JAX API Reference](https://docs.jax.dev/)
*   [JAX Developer Documentation](https://docs.jax.dev/en/latest/developer.html)