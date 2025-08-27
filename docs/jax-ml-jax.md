<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing with Python

**JAX empowers you to write high-performance numerical code in Python, leveraging automatic differentiation, just-in-time compilation, and hardware acceleration for large-scale machine learning and scientific computing.**  [See the original repo](https://github.com/jax-ml/jax)

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions with `jax.grad`, enabling complex optimization tasks.
*   **Just-in-Time (JIT) Compilation:**  Boost performance with `jax.jit` by compiling your functions using XLA, optimizing them for various hardware accelerators.
*   **Automatic Vectorization:**  Simplify your code and enhance performance with `jax.vmap`, which automatically vectorizes functions, enabling efficient computations on arrays.
*   **Hardware Acceleration:**  Run your code efficiently on GPUs, TPUs, and other accelerators for faster computation.
*   **Composable Transformations:**  Combine JAX's transformations (`grad`, `jit`, `vmap`) in powerful ways to build custom, high-performance numerical computations.
*   **Scalable Computing:**  Scale your computations across multiple devices using compiler-based automatic parallelization, explicit sharding, or manual per-device programming.

**Jump to:**  [Transformations](#transformations) | [Scaling](#scaling) | [Installation](#installation) | [Reference Docs](#reference-documentation)

## Transformations

JAX's core strength lies in its ability to transform numerical functions.

### Automatic Differentiation with `grad`

Calculate reverse-mode gradients efficiently:

```python
import jax
import jax.numpy as jnp

def tanh(x):
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = jax.grad(tanh)
print(grad_tanh(1.0))  # Output: 0.4199743
```

Differentiate to any order:

```python
print(jax.grad(jax.grad(jax.grad(tanh)))(1.0))  # Output: 0.62162673
```

Differentiation supports Python control flow.  Explore the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for detailed information.

### Compilation with `jit`

Use XLA for end-to-end function compilation with `@jit` or as a higher-order function:

```python
import jax
import jax.numpy as jnp

def slow_f(x):
  # Element-wise ops see a large benefit from fusion
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jax.jit(slow_f)
# %timeit -n10 -r3 fast_f(x)  # Example timing
# %timeit -n10 -r3 slow_f(x)  # Example timing
```

See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for control flow restrictions.

### Auto-vectorization with `vmap`

Map a function along array axes:

```python
import jax
import jax.numpy as jnp

def l1_distance(x, y):
  assert x.ndim == y.ndim == 1
  return jnp.sum(jnp.abs(x - y))

def pairwise_distances(dist1D, xs):
  return jax.vmap(jax.vmap(dist1D, (0, None)), (None, 0))(xs, xs)

xs = jax.random.normal(jax.random.key(0), (100, 3))
dists = pairwise_distances(l1_distance, xs)
# dists.shape  # (100, 100)
```

Compose `jax.vmap` with `jax.grad` and `jax.jit` for per-example gradients:

```python
# per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

## Scaling

Scale computations across thousands of devices with these approaches:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler handles data sharding.
*   **Explicit sharding and automatic partitioning:** Use explicit sharding within a global view.
*   **Manual per-device programming:** Get a per-device view for fine-grained control.

| Mode         | View?      | Explicit Sharding? | Explicit Collectives? |
|--------------|------------|--------------------|-----------------------|
| Auto         | Global     | ❌                  | ❌                     |
| Explicit     | Global     | ✅                  | ❌                     |
| Manual       | Per-device | ✅                  | ✅                     |

```python
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
mesh = jax.make_mesh((8,), ('data',), axis_types=(AxisType.Explicit,))
set_mesh(mesh)

# parameters are sharded for FSDP:
# for W, b in params:
#  print(f'{jax.typeof(W)}')  # f32[512@data,512]
#  print(f'{jax.typeof(b)}')  # f32[512]

# shard data for batch parallelism:
# inputs, targets = jax.device_put((inputs, targets), P('data'))

# evaluate gradients, automatically parallelized!
# gradfun = jax.jit(jax.grad(loss))
# param_grads = gradfun(params, (inputs, targets))
```

See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more details.

## Gotchas and Sharp Bits

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for common pitfalls.

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

See the [documentation](https://docs.jax.dev/en/latest/installation.html) for alternatives, including source compilation, Docker, conda builds, and FAQs.

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

A paper describing JAX's early capabilities is available at [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf). A more comprehensive paper is forthcoming.

## Reference documentation

For detailed API information, see the [reference documentation](https://docs.jax.dev/).
For JAX development, see the [developer documentation](https://docs.jax.dev/en/latest/developer.html).