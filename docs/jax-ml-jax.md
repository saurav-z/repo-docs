<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX empowers you to write fast, scalable, and differentiable Python code, unlocking new possibilities in scientific computing and machine learning.** [Explore the JAX repository on GitHub](https://github.com/jax-ml/jax).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions using `jax.grad`. Differentiate through loops, branches, recursion, and closures to any order.
*   **Just-In-Time Compilation:** Accelerate your code with XLA compilation using `jax.jit` for optimized performance on CPUs, GPUs, TPUs, and more.
*   **Vectorization:**  Use `jax.vmap` for efficient auto-vectorization, enabling you to write concise and performant code without manual loop management.
*   **Scalability:**  Scale computations across thousands of devices with compiler-based auto-parallelization, explicit sharding, and manual per-device programming.
*   **Composable Transformations:**  Combine automatic differentiation, compilation, and vectorization to create powerful and optimized numerical functions.

## What is JAX?

JAX is a versatile Python library designed for high-performance numerical computing and program transformation. It excels in accelerator-oriented array computation, making it ideal for large-scale machine learning and scientific applications. JAX's core strength lies in its ability to transform numerical functions in powerful ways.

JAX seamlessly integrates with Python and NumPy, offering automatic differentiation, just-in-time compilation, and auto-vectorization capabilities. It leverages XLA (Accelerated Linear Algebra) to optimize code execution on various hardware accelerators like GPUs, TPUs, and other devices.

**Important Note:** JAX is a research project, not an official Google product. Be prepared for [sharp edges](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) and contribute by reporting bugs and providing feedback!

```python
import jax
import jax.numpy as jnp

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jax.jit(jax.grad(loss))  # compiled gradient evaluation function
perex_grads = jax.jit(jax.vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads
```

## Core Transformations

JAX provides a suite of powerful transformations for manipulating numerical functions:

*   **Automatic differentiation with `grad`**: Compute reverse-mode gradients efficiently.
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
    You can differentiate to any order. See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) and the [reference docs on automatic differentiation](https://docs.jax.dev/en/latest/jax.html#automatic-differentiation) for more.

*   **Compilation with `jit`**: Compile your functions end-to-end with XLA for performance gains.
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
    Using `jax.jit` constrains the kind of Python control flow the function can use; see the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for more.

*   **Auto-vectorization with `vmap`**: Vectorize functions along array axes for improved performance.
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
    You can combine `jax.vmap` with `jax.grad` and `jax.jit` for efficient operations such as Jacobian matrices or per-example gradients.

## Scaling

JAX offers various approaches for scaling computations across multiple devices:

*   **Compiler-based automatic parallelization:** Program as if using a single machine; the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Explicitly define data shardings using JAX types for more control.
*   **Manual per-device programming:** Write code with a per-device view of data and computation using explicit collectives.

| Mode       | View?      | Explicit sharding? | Explicit Collectives? |
|------------|------------|--------------------|-----------------------|
| Auto       | Global     | ❌                 | ❌                     |
| Explicit   | Global     | ✅                 | ❌                     |
| Manual     | Per-device | ✅                 | ✅                     |

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
See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for more details.

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

For alternative installation methods, including compiling from source, Docker, conda, and FAQs, consult the [JAX documentation](https://docs.jax.dev/en/latest/installation.html).

## Gotchas and Sharp Bits

Be aware of potential issues; see the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)

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
A more comprehensive and updated paper on JAX's capabilities is forthcoming.
## Reference Documentation

Access the comprehensive [reference documentation](https://docs.jax.dev/) for detailed information on the JAX API. For JAX developer guidance, explore the [developer documentation](https://docs.jax.dev/en/latest/developer.html).