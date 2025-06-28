<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing with Automatic Differentiation and XLA Compilation

JAX is a powerful Python library that transforms numerical computations, enabling automatic differentiation, XLA compilation, and parallelization for high-performance machine learning and scientific computing. **[Explore the JAX repository](https://github.com/jax-ml/jax)!**

*   **Automatic Differentiation:** Effortlessly compute gradients of native Python and NumPy functions, including through loops, branches, recursion, and closures.
*   **XLA Compilation:** Compile your NumPy programs for significant performance gains on GPUs, TPUs, and other hardware accelerators using XLA.
*   **Function Transformations:** Leverage composable function transformations like `jax.grad`, `jax.jit`, and `jax.vmap` to optimize your code.
*   **Scalable Computing:** Scale your computations across thousands of devices using compiler-based automatic parallelization, explicit sharding, and manual per-device programming.
*   **Hardware Acceleration:** Run your code on CPUs, GPUs (NVIDIA, AMD, Apple, Intel), and TPUs for optimized performance.

## Key Features

JAX offers a suite of powerful features to accelerate your numerical computations:

*   **Automatic Differentiation:**
    *   Compute gradients efficiently using `jax.grad`.
    *   Differentiate to any order.
    *   Works with Python control flow.
    *   See the [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) for examples.
*   **Compilation with `jit`:**
    *   Use XLA to compile functions for performance.
    *   Optimize element-wise operations.
    *   See the tutorial on [Control Flow and Logical Operators with JIT](https://docs.jax.dev/en/latest/control-flow.html) for details on control flow limitations.
*   **Auto-vectorization with `vmap`:**
    *   Apply functions along array axes.
    *   Avoid explicit looping for improved performance.
*   **Scaling:**
    *   Use compiler-based automatic parallelization.
    *   Explicit sharding and automatic partitioning.
    *   Manual per-device programming.
    *   See the [tutorial](https://docs.jax.dev/en/latest/sharded-computation.html) and
        [advanced guides](https://docs.jax.dev/en/latest/advanced_guide.html) for scaling documentation.

## Getting Started

### Installation

JAX supports various platforms. Here's a quick guide:

#### Supported Platforms

|            | Linux x86_64 | Linux aarch64 | Mac aarch64  | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a            | n/a                 |
| AMD GPU    | yes          | no            | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | n/a            | n/a                 |
| Intel GPU  | experimental | n/a           | n/a          | no             | no                  |

#### Installation Instructions

| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                     |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                      |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).  |

For more detailed instructions, see the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html).

### Example: Simple Gradient Calculation

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

## More Resources

*   **[Reference Documentation](https://docs.jax.dev/):**  Explore the comprehensive JAX API.
*   **[Developer Documentation](https://docs.jax.dev/en/latest/developer.html):** Get started as a JAX developer.
*   **[Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html):** Be aware of common pitfalls.

## Citing JAX

To cite this repository, use the following BibTeX entry:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```