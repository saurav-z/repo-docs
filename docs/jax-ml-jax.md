<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo"></img>
</div>

# JAX: High-Performance Numerical Computing with Python

JAX empowers you to write fast and efficient numerical code, transforming your Python and NumPy programs with ease.  Check out the original repository [here](https://github.com/jax-ml/jax).

**Key Features:**

*   **Automatic Differentiation:** Effortlessly calculate gradients of your Python and NumPy code, enabling complex machine learning and scientific computations.
*   **Just-in-Time (JIT) Compilation:** Speed up your code with XLA compilation, optimizing performance on GPUs, TPUs, and other hardware accelerators.
*   **Vectorization (vmap):**  Transform functions to operate on batches of data, streamlining your code and improving efficiency.
*   **Scalability:** Scale your computations across multiple devices, utilizing compiler-based automatic parallelization and explicit sharding.
*   **Composable Transformations:**  Combine automatic differentiation, JIT compilation, and vectorization for powerful and flexible numerical computations.

## Table of Contents

*   [What is JAX?](#what-is-jax)
    *   [Transformations](#transformations)
        *   [Automatic differentiation with `grad`](#automatic-differentiation-with-grad)
        *   [Compilation with `jit`](#compilation-with-jit)
        *   [Auto-vectorization with `vmap`](#auto-vectorization-with-vmap)
    *   [Scaling](#scaling)
    *   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
    *   [Installation](#installation)
        *   [Supported platforms](#supported-platforms)
        *   [Instructions](#instructions)
    *   [Citing JAX](#citing-jax)
    *   [Reference documentation](#reference-documentation)

## What is JAX?

JAX is a versatile Python library designed for high-performance numerical computing and machine learning.  It provides powerful tools for transforming and optimizing your code.

At its core, JAX is an extensible system for transforming numerical functions.

### Transformations

JAX offers a set of core transformations to manipulate functions.

#### Automatic differentiation with `grad`

Easily compute reverse-mode gradients with `jax.grad`:

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

Differentiation can be applied to any order and with Python control flow.

#### Compilation with `jit`

Use `jax.jit` (just-in-time compilation) to compile functions for optimized performance.

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

#### Auto-vectorization with `vmap`

Apply functions along array axes with `jax.vmap`.

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

JAX supports scaling computations across thousands of devices through compiler-based automatic parallelization, explicit sharding, and manual per-device programming.

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

### Gotchas and Sharp Bits

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for more information.

### Installation

#### Supported platforms

JAX supports various platforms:

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

#### Instructions

To install JAX, follow these instructions:

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for other installation options.

### Citing JAX

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

### Reference documentation

For detailed information on the JAX API, see the [reference documentation](https://docs.jax.dev/).
For JAX developer information, please consult the [developer documentation](https://docs.jax.dev/en/latest/developer.html).