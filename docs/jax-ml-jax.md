<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing with Python

**JAX empowers you to perform complex numerical computations at scale with automatic differentiation, just-in-time compilation, and hardware acceleration.**  Check out the original [JAX repository](https://github.com/jax-ml/jax).

*   **Automatic Differentiation:** Easily compute gradients of native Python and NumPy functions, including those with loops, branches, and recursion.
*   **Just-in-Time (JIT) Compilation:** Accelerate your code with XLA compilation for TPUs, GPUs, and other accelerators.
*   **Vectorization (vmap):** Efficiently map functions across array axes, enabling faster and more concise code,
*   **Composable Transformations:** Build powerful numerical programs using the core transformations like `jax.grad`, `jax.jit`, and `jax.vmap`.
*   **Scalable Computations:** Scale your computations across thousands of devices using compiler-based automatic parallelization and explicit sharding.
*   **Hardware Acceleration:** Run your numerical code on CPUs, GPUs (NVIDIA, AMD, Apple, Intel), and TPUs for optimal performance.

## Key Features

JAX provides powerful tools for numerical computation and machine learning:

*   **Automatic Differentiation:** Calculate derivatives of complex functions with ease, supporting reverse-mode and forward-mode differentiation.
*   **JIT Compilation with XLA:** Optimize your code for speed by compiling NumPy programs for various hardware accelerators (TPUs, GPUs).
*   **Vectorization (vmap):** Eliminate the need for manual looping and batch dimensions for better performance.
*   **Scaling Capabilities:** Distribute your computations across numerous devices using different parallelization strategies.

## Transformations

JAX's core strength lies in its ability to transform numerical functions. Here are the main transformation functions:

*   **`jax.grad`:**  Compute gradients efficiently for automatic differentiation.
*   **`jax.jit`:**  Compile functions with XLA for accelerated performance.
*   **`jax.vmap`:**  Vectorize functions for parallel execution.

## Scaling

JAX offers multiple approaches to scale your computations:

*   **Compiler-Based Automatic Parallelization:**  Automatically shard data and partition computation.
*   **Explicit Sharding:**  Specify data sharding using JAX types for fine-grained control.
*   **Manual Per-Device Programming:**  Write code specifically for each device with explicit communication.

## Installation

JAX supports various platforms for CPU, GPU, and TPU usage.

### Supported Platforms:

| Platform        | CPU | NVIDIA GPU | Google TPU | AMD GPU | Apple GPU | Intel GPU | Windows |
|-----------------|-----|------------|------------|---------|-----------|-----------|---------|
| Linux x86_64    | yes | yes        | yes        | yes     | no        | experimental   | yes     |
| Linux aarch64   | yes | yes        | no         | no      | no        | no        | no      |
| Mac aarch64     | yes | no         | no         | no      | experimental | no        | no      |
| Windows x86_64  | yes | no        | no         | no      | no        | no        | experimental   |
| Windows WSL2 x86_64 | yes | experimental        | no         | no      | no        | no       |  |

### Installation Instructions:

CPU: `pip install -U jax`

NVIDIA GPU: `pip install -U "jax[cuda12]"`

Google TPU: `pip install -U "jax[tpu]"`

AMD GPU (Linux): Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).

Mac GPU: Follow [Apple's instructions](https://developer.apple.com/metal/jax/).

Intel GPU: Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

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

Consult the [reference documentation](https://docs.jax.dev/) and [developer documentation](https://docs.jax.dev/en/latest/developer.html) for detailed information on the JAX API and contributing to the project.

## Gotchas

See the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).