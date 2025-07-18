<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

JAX is a powerful Python library that transforms NumPy programs for high-performance numerical computation and large-scale machine learning.  [Explore the JAX repository](https://github.com/jax-ml/jax/).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   [**Transformations**](#transformations)
*   [**Scaling**](#scaling)
*   [**Install Guide**](#installation)
*   [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features of JAX

*   **Automatic Differentiation:**  Effortlessly compute gradients of native Python and NumPy functions, including derivatives of derivatives.
*   **Just-In-Time Compilation (JIT):**  Compile Python and NumPy code for optimized performance on accelerators like GPUs, TPUs, and CPUs using XLA.
*   **Vectorization:**  Automatically vectorize functions for efficient processing of batches of data with `jax.vmap`.
*   **Composable Transformations:** Build complex computations by combining JAX's transformations like `jax.grad`, `jax.jit`, and `jax.vmap`.
*   **Scalability:**  Scale your computations across thousands of devices through compiler-based automatic parallelization, explicit sharding, and manual per-device programming.

## Transformations

JAX offers powerful function transformations as its core, allowing you to manipulate numerical functions in various ways:

### Automatic Differentiation with `grad`

Use `jax.grad` to compute reverse-mode gradients efficiently.  This allows you to calculate gradients of Python and NumPy functions, even through control flow and recursion.

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

You can also compute higher-order derivatives:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

For further details, consult the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [automatic differentiation reference docs](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Employ XLA to compile your functions with `jax.jit` for improved performance. This compiles functions end-to-end, leading to optimizations and faster execution.

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

For more information, see the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html).

### Auto-vectorization with `vmap`

Use `jax.vmap` to automatically vectorize your functions, allowing them to operate on arrays along specified axes.  This effectively pushes the loop down to the function’s primitive operations, improving performance.

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

You can combine `jax.vmap` with other transformations such as `jax.grad` and `jax.jit` for advanced capabilities like per-example gradients.

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX allows you to scale your computations across numerous devices using the following modes:

*   **Compiler-based automatic parallelization**: Program with a global view, letting the compiler handle data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning**: Maintain a global view but explicitly define data shardings using JAX types.
*   **Manual per-device programming**: Access a per-device view of data and computation, utilizing explicit collectives for communication.

| Mode        | View?    | Explicit sharding? | Explicit Collectives? |
| ----------- | -------- | ------------------ | --------------------- |
| Auto        | Global   | ❌                 | ❌                    |
| Explicit    | Global   | ✅                 | ❌                    |
| Manual      | Per-device | ✅                 | ✅                    |

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

For further information, see the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html).

## Gotchas and Sharp Edges

Be aware of the potential [Gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) when using JAX.

## Installation

### Supported Platforms

JAX supports various platforms:

| Platform        | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
| --------------- | ------------ | ------------- | ------------ | -------------- | ------------------- |
| CPU             | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU      | yes          | yes           | n/a          | no             | experimental        |
| Google TPU      | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU         | yes          | no            | n/a          | no             | no                  |
| Apple GPU       | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU       | experimental | n/a           | n/a          | no             | no                  |

### Instructions

Follow the instructions for your platform:

| Platform        | Instructions                                                                                                    |
| --------------- | -----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

Refer to the [JAX documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation strategies, including compiling from source, using Docker, and addressing common installation questions.

## Citing JAX

To cite this project, use the following:

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

Find comprehensive details about the JAX API in the [reference documentation](https://docs.jax.dev/).
Also, check out the [developer documentation](https://docs.jax.dev/en/latest/developer.html) for starting your JAX development.