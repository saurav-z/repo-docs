<div align="center">
<img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: High-Performance Numerical Computing with Python

JAX is a powerful Python library for accelerating array computations and program transformations, ideal for high-performance numerical computing and large-scale machine learning. Explore the possibilities and learn more at the [original repo](https://github.com/jax-ml/jax).

[![Continuous integration](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/jax-ml/jax/actions/workflows/ci-build.yaml)
[![PyPI version](https://img.shields.io/pypi/v/jax)](https://pypi.org/project/jax/)

*   [**Transformations**](#transformations)
*   [**Scaling**](#scaling)
*   [**Installation Guide**](#installation)
*   [**Change Logs**](https://docs.jax.dev/en/latest/changelog.html)
*   [**Reference Docs**](https://docs.jax.dev/en/latest/)

## What is JAX?

JAX transforms Python and NumPy code for high-performance numerical computing. It provides automatic differentiation, just-in-time compilation, and auto-vectorization, enabling researchers and developers to scale computations across various hardware accelerators like TPUs, GPUs, and more. JAX differentiates through loops, branches, recursion, and closures, and supports reverse and forward-mode differentiation that can be composed to any order.

### Key Features

*   **Automatic Differentiation:** Easily compute gradients of native Python and NumPy functions using `jax.grad`.
*   **Just-In-Time Compilation:** Compile and optimize your NumPy programs using `jax.jit` for accelerated execution.
*   **Auto-Vectorization:** Utilize `jax.vmap` to automatically vectorize functions and map them across array axes for improved performance.
*   **Hardware Acceleration:** Leverage XLA (Accelerated Linear Algebra) to run your code on TPUs, GPUs, and other hardware accelerators.
*   **Composable Transformations:** Combine differentiation, compilation, and vectorization to create highly efficient computational workflows.

### Quick Example: Automatic Differentiation

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

### Contents

*   [Transformations](#transformations)
*   [Scaling](#scaling)
*   [Gotchas and Sharp Bits](#gotchas-and-sharp-bits)
*   [Installation](#installation)
*   [Neural Network Libraries](#neural-network-libraries)
*   [Citing JAX](#citing-jax)
*   [Reference Documentation](#reference-documentation)

## Transformations

JAX's core strength lies in its ability to transform numerical functions. Key transformations include:

### Automatic Differentiation with `grad`

Efficiently calculate reverse-mode gradients using `jax.grad`. It works with various Python control flow structures.

### Compilation with `jit`

Compile functions end-to-end with XLA using `jax.jit` for significant performance gains, especially with element-wise operations.

### Auto-vectorization with `vmap`

Apply functions along array axes using `jax.vmap`, efficiently handling batch processing and reducing the need for explicit loop management.

## Scaling

JAX facilitates scaling computations across multiple devices with three main approaches:

*   **Compiler-based automatic parallelization:** Program as if using a single global machine, and the compiler handles data sharding and computation partitioning.
*   **Explicit sharding and automatic partitioning:** Define explicit shardings using JAX types while retaining a global view.
*   **Manual per-device programming:** Access data and computation with a per-device view, including explicit collectives.

| Mode         | View?      | Explicit Sharding? | Explicit Collectives? |
|--------------|------------|--------------------|-----------------------|
| Auto         | Global     | ❌                  | ❌                     |
| Explicit     | Global     | ✅                  | ❌                     |
| Manual       | Per-device | ✅                  | ✅                     |

## Gotchas and Sharp Bits

Refer to the [Gotchas Notebook](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for potential pitfalls and considerations when using JAX.

## Installation

### Supported Platforms

JAX offers broad platform support:

|            | Linux x86\_64 | Linux aarch64 | Mac aarch64  | Windows x86\_64 | Windows WSL2 x86\_64 |
|------------|---------------|---------------|--------------|-----------------|---------------------|
| CPU        | yes           | yes           | yes          | yes             | yes                 |
| NVIDIA GPU | yes           | yes           | n/a          | no              | experimental        |
| Google TPU | yes           | n/a           | n/a          | n/a             | n/a                 |
| AMD GPU    | yes           | no            | n/a          | no              | no                  |
| Apple GPU  | n/a           | no            | experimental | n/a             | n/a                 |
| Intel GPU  | experimental  | n/a           | n/a          | no              | no                  |

### Installation Instructions

Install JAX based on your platform and hardware:

| Platform        | Instructions                                                                                                       |
|-----------------|--------------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                               |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                     |
| Google TPU      | `pip install -U "jax[tpu]"`                                                                                        |
| AMD GPU (Linux) | Follow [AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).                         |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                             |
| Intel GPU       | Follow [Intel's instructions](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).   |

For alternative installation methods, consult the [documentation](https://docs.jax.dev/en/latest/installation.html).

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

A research paper related to the early JAX version is available at [SysML 2018](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).

## Reference Documentation

Explore the detailed [reference documentation](https://docs.jax.dev/) for API information and the [developer documentation](https://docs.jax.dev/en/latest/developer.html) to get started with JAX development.