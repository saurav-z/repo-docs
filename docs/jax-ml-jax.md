<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="JAX Logo">
</div>

# JAX: High-Performance Numerical Computing and Machine Learning in Python

**JAX empowers you to write efficient and scalable numerical programs with automatic differentiation, just-in-time compilation, and array transformations.** Learn more and contribute on [GitHub](https://github.com/jax-ml/jax).

**Key Features:**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions with `jax.grad`.
*   **Just-in-Time (JIT) Compilation:** Accelerate your code with XLA using `jax.jit` for optimized performance on CPUs, GPUs, and TPUs.
*   **Array Transformations:** Utilize `jax.vmap` for automatic vectorization and efficient operations on arrays.
*   **Scalable Computing:** Scale your computations across thousands of devices with compiler-based parallelization, explicit sharding, and manual per-device programming.
*   **Composable Transformations:** Combine automatic differentiation, JIT compilation, and vectorization for powerful and flexible numerical computations.

**Table of Contents:**

*   [What is JAX?](#what-is-jax)
    *   [Transformations](#transformations)
        *   [Automatic Differentiation with `grad`](#automatic-differentation-with-grad)
        *   [Compilation with `jit`](#compilation-with-jit)
        *   [Auto-vectorization with `vmap`](#auto-vectorization-with-vmap)
    *   [Scaling](#scaling)
    *   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
    *   [Installation](#installation)
        *   [Supported Platforms](#supported-platforms)
        *   [Instructions](#instructions)
    *   [Citing JAX](#citing-jax)
    *   [Reference Documentation](#reference-documentation)

## What is JAX?

JAX is a Python library designed for high-performance numerical computing and large-scale machine learning. It combines the flexibility of Python and NumPy with powerful program transformations, enabling you to write efficient and scalable code for accelerator-oriented array computation. JAX excels at automatic differentiation, just-in-time compilation, and array vectorization, making it a versatile tool for researchers and developers alike.

### Transformations

JAX's core strength lies in its ability to transform numerical functions. Key transformations include automatic differentiation, just-in-time compilation, and auto-vectorization.

#### Automatic Differentiation with `grad`

JAX simplifies the process of calculating gradients using `jax.grad`. It supports reverse-mode differentiation, allowing you to compute gradients of complex functions, including those with loops, branches, recursion, and closures.

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

#### Compilation with `jit`

JAX uses XLA to compile your functions for optimized execution on CPUs, GPUs, and TPUs using `jax.jit`. This can significantly speed up your code by fusing operations and reducing overhead.

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

`jax.vmap` enables automatic vectorization, allowing you to efficiently apply a function to multiple inputs. It pushes the loop down to the function's primitive operations for improved performance.

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

`vmap` can be composed with `jax.grad` and `jax.jit` for efficient gradient calculations, such as per-example gradients:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
```

### Scaling

JAX offers multiple approaches to scale your computations across thousands of devices:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler shards and partitions data and computation.
*   **Explicit sharding and automatic partitioning:** Use a global view with explicit data shardings, inspectable using `jax.typeof`.
*   **Manual per-device programming:** Work with a per-device view of data and computation, enabling communication with explicit collectives.

### Gotchas and Sharp Bits

Be aware of potential "gotchas" and limitations. See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for more information.

### Installation

#### Supported Platforms

| Platform         | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|-----------------|--------------|---------------|--------------|----------------|---------------------|
| CPU             | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU      | yes          | yes           | n/a          | no             | experimental        |
| Google TPU      | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU         | yes          | no            | n/a          | no             | no                  |
| Apple GPU       | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU       | experimental | n/a           | n/a          | no             | no                  |

#### Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

For detailed installation instructions, including alternative methods like compiling from source, Docker, and conda, refer to the [documentation](https://docs.jax.dev/en/latest/installation.html).

### Citing JAX

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

### Reference Documentation

Explore the complete JAX API in the [reference documentation](https://docs.jax.dev/). Developers can find useful information in the [developer documentation](https://docs.jax.dev/en/latest/developer.html).