<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: Accelerate Your Numerical Computing with Composable Transformations

[JAX](https://github.com/jax-ml/jax) is a powerful Python library designed for high-performance numerical computing and machine learning, offering automatic differentiation, just-in-time compilation, and vectorization capabilities.

*   **Automatic Differentiation:** Effortlessly compute gradients of complex functions, including those with loops, branches, and recursion.
*   **Just-In-Time Compilation (JIT):** Compile your Python and NumPy code with XLA for optimized performance on GPUs, TPUs, and other accelerators.
*   **Vectorization (vmap):**  Efficiently vectorize functions to work on batches of data, simplifying your code and boosting performance.
*   **Scaling:** Easily scale computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.

## Key Features

*   **Composable Function Transformations:** JAX excels at transforming numerical functions, enabling advanced capabilities such as automatic differentiation, just-in-time compilation, and vectorization.
*   **Automatic Differentiation with `grad`:** Quickly calculate reverse-mode gradients (backpropagation) with the `jax.grad` function.  Differentiate through Python control flow and compose gradients to any order.
*   **Compilation with `jit`:** Compile functions using XLA via `jax.jit` to optimize performance by fusing operations for accelerators like GPUs and TPUs.
*   **Auto-vectorization with `vmap`:** Utilize `jax.vmap` to automatically vectorize functions, making it easy to operate on batches of data, saving you from managing batch dimensions manually, and improving performance.
*   **Scaling Across Devices:** Scale your computations using auto, explicit, or manual parallelization methods.

## Transformations

JAX provides an extensible system for transforming numerical functions. Key transformations include:

### Automatic Differentiation with `grad`

Compute gradients efficiently using `jax.grad`.  See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more details.

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

Use XLA to compile your functions with `jit`.  See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more.

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

Vectorize functions with `vmap` to work on array axes.

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

Scale your computations across thousands of devices using:
*   **Compiler-based automatic parallelization**
*   **Explicit sharding and automatic partitioning**
*   **Manual per-device programming**

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

## Gotchas and Sharp Edges

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

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

See [the documentation](https://docs.jax.dev/en/latest/installation.html) for other installation methods.

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

*   [Reference documentation](https://docs.jax.dev/): Comprehensive API details.
*   [Developer documentation](https://docs.jax.dev/en/latest/developer.html): For developers.