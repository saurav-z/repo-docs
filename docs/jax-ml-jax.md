<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing with Automatic Differentiation and XLA

**JAX is a powerful Python library designed for high-performance numerical computing and machine learning, offering automatic differentiation, XLA compilation, and array transformations.**  Explore the [JAX GitHub repository](https://github.com/jax-ml/jax) to learn more.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   [**Transformations**](#transformations)
*   [**Scaling**](#scaling)
*   [**Installation Guide**](#installation)
*   [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)

## Key Features of JAX

*   **Automatic Differentiation:**  Compute gradients of native Python and NumPy functions effortlessly, with support for reverse-mode (backpropagation), forward-mode, and higher-order derivatives.
*   **XLA Compilation:** Leverage XLA (Accelerated Linear Algebra) to compile and optimize your NumPy programs for CPUs, GPUs, TPUs, and other hardware accelerators, boosting performance.
*   **Function Transformations:** JAX provides powerful function transformations like `jax.grad` (for differentiation), `jax.jit` (for compilation), and `jax.vmap` (for auto-vectorization), enabling composable and efficient numerical computation.
*   **Scalability:**  Scale your computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Python-Friendly:** Integrate seamlessly with your existing Python and NumPy code.

## Transformations

JAX provides a suite of function transformations at its core, including:

### Automatic Differentiation with `grad`

Use `jax.grad` to compute gradients efficiently.

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
Differentiation also works with Python control flow.

See the [JAX Autodiff
Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
and the [reference docs on automatic
differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation)
for more.

### Compilation with `jit`

Compile functions with XLA using `jax.jit`.

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
See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)
for more.

### Auto-vectorization with `vmap`

Use `jax.vmap` to map a function along array axes.

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

You can combine `jax.vmap` with `jax.grad` and `jax.jit`.

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

Scale your computations across various devices using:

*   **Compiler-based automatic parallelization:**  The compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:**  Use explicit shardings in JAX types.
*   **Manual per-device programming:**  Per-device view of data and computation with explicit collectives.

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

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and
[advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Bits

See the [Gotchas
Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported Platforms

| Platform        | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|-----------------|--------------|---------------|--------------|----------------|---------------------|
| CPU             | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU      | yes          | yes           | n/a          | no             | experimental        |
| Google TPU      | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU         | yes          | no            | n/a          | no             | no                  |
| Apple GPU       | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU       | experimental | n/a           | n/a          | no             | no                  |

### Installation Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

Refer to [the documentation](https://docs.jax.dev/en/latest/installation.html) for alternative installation strategies, including building from source and using Docker, as well as for answers to frequently asked questions.

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

## Reference Documentation

For detailed information about the JAX API, see the
[reference documentation](https://docs.jax.dev/).
For getting started as a JAX developer, see the
[developer documentation](https://docs.jax.dev/en/latest/developer.html).