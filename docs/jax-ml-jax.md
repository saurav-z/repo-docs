<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Program Transformation

[**JAX**](https://github.com/jax-ml/jax) is a powerful Python library that brings the performance of accelerators to your NumPy programs while enabling automatic differentiation and advanced program transformations.

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

## Key Features of JAX

*   **Automatic Differentiation:**  Effortlessly compute gradients of native Python and NumPy functions, including differentiation through loops, branches, recursion, and closures.
*   **Just-In-Time (JIT) Compilation:**  Compile your NumPy programs for optimized performance on TPUs, GPUs, and other hardware accelerators using XLA.
*   **Vectorization with `vmap`:**  Efficiently vectorize functions across array axes for improved performance and simplified code.
*   **Composable Transformations:** Build complex numerical computations by composing the powerful `grad`, `jit`, and `vmap` transformations.
*   **Scalable Computing:**  Scale computations across thousands of devices using automatic parallelization, explicit sharding, and per-device programming.

## Core Capabilities

JAX's core strength lies in its ability to transform numerical functions. It provides three primary transformations:

### Automatic Differentiation with `grad`

Effortlessly compute reverse-mode gradients, to any order, even through Python control flow.

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

For more information, see the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation).

### Compilation with `jit`

Compile your functions end-to-end to optimize performance with the `@jit` decorator or as a higher-order function.

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

Apply a function to array axes with [`vmap`](https://docs.jax.dev/en/latest/jax.html#vectorization-vmap). Instead of looping over function applications, it pushes the loop down onto the function's primitive operations.

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

By composing `jax.vmap` with `jax.grad` and `jax.jit`, we can get efficient Jacobian matrices, or per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

JAX offers multiple approaches for scaling your computations across thousands of devices:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, allowing the compiler to handle data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Maintain a global view while explicitly defining data shardings within JAX types.
*   **Manual per-device programming:**  Work with a per-device view of data and computation, with explicit communication primitives.

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more.

## Gotchas and Sharp Edges

Be aware of potential [Gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Installation

### Supported platforms

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

For alternative installation strategies, see the [documentation](https://docs.jax.dev/en/latest/installation.html).

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

For a related paper, see the [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference documentation

For detailed information on the JAX API and development, please refer to the [reference documentation](https://docs.jax.dev/).