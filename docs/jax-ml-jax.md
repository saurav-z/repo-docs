<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Machine Learning with Python

[**Transformations**](#transformations) | [**Scaling**](#scaling) | [**Installation**](#installation) | [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html) | [**Reference Docs**](https://docs.jax.dev/en/latest/)

**JAX** is a powerful Python library that brings the flexibility of NumPy to modern accelerators, enabling automatic differentiation, just-in-time compilation, and efficient scaling for numerical computing and machine learning.  [Explore the JAX Repository](https://github.com/jax-ml/jax).

## Key Features of JAX:

*   **Automatic Differentiation:**  Effortlessly compute gradients of native Python and NumPy functions, including those with loops, branches, recursion, and closures, with support for arbitrary-order derivatives via `jax.grad`.
*   **Just-in-Time (JIT) Compilation:**  Compile Python and NumPy code using XLA for optimized performance on TPUs, GPUs, and other hardware accelerators with `jax.jit`.
*   **Vectorization and Parallelization:**  Efficiently vectorize functions across array axes with `jax.vmap` and scale computations across thousands of devices using compiler-based auto-parallelization, explicit sharding, and manual per-device programming.
*   **Composable Transformations:** Build complex, optimized computational pipelines by composing transformations like `jax.grad`, `jax.jit`, and `jax.vmap`.
*   **Extensible System:** Design your own pure functions with an extensible system for composable function transformations at scale.

## Core Transformations

JAX's core strength lies in its ability to transform numerical functions, providing powerful tools for developers.

### Automatic Differentiation with `grad`

Compute reverse-mode gradients efficiently:

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

Differentiate to any order:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))
# prints 0.62162673
```

Differentiate through Python control flow:

```python
def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = jax.grad(abs_val)
print(abs_val_grad(1.0))   # prints 1.0
print(abs_val_grad(-1.0))  # prints -1.0 (abs_val is re-evaluated)
```

Learn more at the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [automatic differentiation reference docs](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Use XLA to compile functions end-to-end for optimized performance:

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

Refer to the [Control Flow and Logical Operators with JIT tutorial](https://docs.jax.dev/en/latest/control-flow.html)

### Auto-vectorization with `vmap`

Map functions along array axes, pushing loops down to primitive operations:

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

Compose `jax.vmap` with `jax.grad` and `jax.jit` for efficient Jacobian matrices, or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling Your Computations

JAX offers multiple strategies for scaling computations across thousands of devices:

*   **Compiler-based automatic parallelization**: Simplifies programming as if on a single global machine, letting the compiler manage data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning**: Uses a global view with explicitly defined data shardings via JAX types.
*   **Manual per-device programming**: Provides per-device views of data and computation with explicit collectives for fine-grained control.

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

For further information, see the [sharded computation tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and the [advanced guide](https://docs.jax.dev/en/latest/advanced_guide.html).

## Gotchas and Sharp Edges

Be aware of the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for known limitations and best practices.

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

Consult the [installation documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation methods like compiling from source, using Docker, and answers to frequently asked questions.

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

## Reference Documentation

For detailed API information, refer to the [reference documentation](https://docs.jax.dev/).

For developer-specific information, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).