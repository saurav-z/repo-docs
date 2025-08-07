html
<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX is a powerful Python library that transforms numerical functions for high-performance computing, offering automatic differentiation, just-in-time compilation, and vectorization.** Explore the official [JAX GitHub Repository](https://github.com/jax-ml/jax) for more details.

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions using `jax.grad`, supporting arbitrary order derivatives and control flow.
*   **Just-in-Time (JIT) Compilation:** Compile your Python and NumPy programs with XLA for significant performance gains on TPUs, GPUs, and other accelerators using `jax.jit`.
*   **Vectorization (vmap):**  Apply functions along array axes for efficient processing, transforming matrix-vector operations into matrix-matrix operations.
*   **Scalable Computations:**  Scale your computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, or manual per-device programming.

**Quick Links:**

*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Installation](#installation)
*   [Reference Documentation](https://docs.jax.dev/)
*   [Change Logs](https://docs.jax.dev/en/latest/changelog.html)

## Transformations

JAX's core strength lies in transforming numerical functions.

### Automatic Differentiation with `grad`

Efficiently calculate reverse-mode gradients:

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

Differentiate to any order and use with Python control flow.

Explore the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Compile functions end-to-end using XLA for faster execution:

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

Learn more about [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for further details.

### Auto-vectorization with `vmap`

Map functions along array axes:

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient operations.

## Scaling

JAX empowers you to scale computations across thousands of devices, offering diverse modes:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine.
*   **Explicit sharding and automatic partitioning:** Global view with explicit data shardings.
*   **Manual per-device programming:** Per-device data and computation views with explicit collectives.

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

Explore the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Bits

Refer to the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for important considerations.

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

See the [documentation](https://docs.jax.dev/en/latest/installation.html) for more installation options.

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

For in-depth details about the JAX API, consult the [reference documentation](https://docs.jax.dev/).