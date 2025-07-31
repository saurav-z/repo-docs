html
<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing with Automatic Differentiation and Compilation

<p>JAX is a powerful Python library for high-performance numerical computing, machine learning, and scientific research, offering automatic differentiation, XLA compilation, and array transformations.</p>

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

[**Key Features**](#key-features) | [**Transformations**](#transformations) | [**Scaling**](#scaling) | [**Installation**](#installation) | [**Documentation**](https://docs.jax.dev/en/latest/)

## Key Features

*   **Automatic Differentiation:** Effortlessly compute gradients of Python and NumPy functions with `jax.grad`.
*   **XLA Compilation:** Compile your numerical programs for accelerated performance on GPUs, TPUs, and other hardware using `jax.jit`.
*   **Function Transformations:** Compose powerful transformations like `jax.grad`, `jax.jit`, and `jax.vmap` for efficient and scalable computations.
*   **Array Processing:** Built on top of XLA, JAX provides efficient array operations, optimized for hardware accelerators.
*   **Vectorization:** Utilize `jax.vmap` for automatic vectorization, improving performance by pushing loops to the primitive operations level.
*   **Scalable Computing:** Scale your computations across thousands of devices with compiler-based auto-parallelization, explicit sharding, or manual per-device programming.

## Transformations

JAX provides a set of composable function transformations that are at the core of its power:

### Automatic Differentiation with `grad`

Compute reverse-mode gradients efficiently using `jax.grad`.

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

You can differentiate to any order with `grad`:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

### Compilation with `jit`

Compile your functions end-to-end with XLA using `jax.jit` for significant performance gains.

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

Apply a function along array axes using `jax.vmap` for efficient batch processing.

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

By composing `jax.vmap` with `jax.grad` and `jax.jit`, we can get efficient
Jacobian matrices, or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX lets you scale your computations efficiently across multiple devices. You can use:

*   **Compiler-based automatic parallelization:** Program as if using a single machine; the compiler handles data sharding.
*   **Explicit sharding and automatic partitioning:** Specify shardings using JAX types.
*   **Manual per-device programming:** Have per-device data/computation views with explicit collectives.

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

For more details, see the complete installation guide in the [JAX documentation](https://docs.jax.dev/en/latest/installation.html).

## Gotchas and Sharp Bits

Be aware of potential limitations and common issues. See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Citing JAX

If you use JAX in your research, please cite it using the following BibTeX entry:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

## Learn More

*   [**Reference Documentation**](https://docs.jax.dev/): Explore the complete JAX API.
*   [**Developer Documentation**](https://docs.jax.dev/en/latest/developer.html):  Get started as a JAX developer.
*   [**Original Repository**](https://github.com/jax-ml/jax): Explore the source code and contribute.