# Transformer Engine: Accelerate Transformer Models with FP8 Precision

Transformer Engine is a powerful library designed to significantly accelerate the training and inference of Transformer models on NVIDIA GPUs, delivering improved performance and reduced memory usage.  [Visit the original repository](https://github.com/NVIDIA/TransformerEngine)

**Key Features:**

*   **FP8 Precision Support:** Enables faster training and inference using 8-bit floating point (FP8) on NVIDIA Hopper, Ada, and Blackwell GPUs.
*   **Optimized Building Blocks:** Provides highly optimized modules and kernels for popular Transformer architectures.
*   **Framework Agnostic C++ API:** Integrates with deep learning libraries to enable FP8 support.
*   **Simplified Mixed Precision:**  Includes APIs and modules for seamless mixed-precision training, with automatic scaling and management of FP8 values.
*   **Broad Hardware Support:** Optimized for NVIDIA Blackwell, Hopper, Ada, and Ampere GPUs.
*   **Integration with Popular Frameworks:** Works with PyTorch, JAX, DeepSpeed, Hugging Face Accelerate, and more.

## Table of Contents

*   [Quickstart](#quickstart)
*   [Installation](#installation)
    *   [System Requirements](#system-requirements)
    *   [Installation Methods](#installation-methods)
        *   [Docker (Recommended)](#docker-recommended)
        *   [pip Installation](#pip-installation)
        *   [conda Installation](#conda-installation)
        *   [Source Installation](#source-installation)
    *   [Environment Variables](#environment-variables)
    *   [Compiling with FlashAttention](#compiling-with-flashattention)
*   [Troubleshooting](#troubleshooting)
*   [Breaking Changes](#breaking-changes)
*   [FP8 Convergence](#fp8-convergence)
*   [Integrations](#integrations)
*   [Contributing](#contributing)
*   [Papers](#papers)
*   [Videos](#videos)
*   [Previous News](#previous-news)

## Quickstart

Get started quickly with the Transformer Engine using the provided examples for PyTorch and JAX.

**PyTorch Example:**

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(inp)

loss = out.sum()
loss.backward()
```

**JAX Example (Flax):**

```python
import flax
import jax
import jax.numpy as jnp
import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.common import recipe

BATCH = 32
SEQLEN = 128
HIDDEN = 1024

# Initialize RNG and inputs.
rng = jax.random.PRNGKey(0)
init_rng, data_rng = jax.random.split(rng)
inp = jax.random.normal(data_rng, [BATCH, SEQLEN, HIDDEN], jnp.float32)

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID)

# Enable autocasting for the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    model = te_flax.DenseGeneral(features=HIDDEN)

    def loss_fn(params, other_vars, inp):
        out = model.apply({'params':params, **other_vars}, inp)
        return jnp.mean(out)

    # Initialize models.
    variables = model.init(init_rng, inp)
    other_variables, params = flax.core.pop(variables, 'params')

    # Construct the forward and backward function
    fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

    for _ in range(10):
        loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)
```

For a more comprehensive tutorial, check out our [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **Software:**
    *   CUDA: 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
    *   cuDNN: 9.3+
    *   Compiler: GCC 9+ or Clang 10+ with C++17 support
    *   Python: 3.12 recommended
*   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

The easiest way to get started is with Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

For example:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
```

Where 25.08 (corresponding to August 2025 release) is the container version.

**Benefits of using NGC containers:**

*   All dependencies are pre-installed with compatible versions and optimized configurations.
*   NGC PyTorch 23.08+ containers include FlashAttention-2.

#### pip Installation

**Prerequisites for pip installation:**

*   A compatible C++ compiler
*   CUDA Toolkit with cuDNN and NVCC (NVIDIA CUDA Compiler) installed

Install the latest stable version:

```bash
# For PyTorch integration
pip install --no-build-isolation transformer_engine[pytorch]

# For JAX integration
pip install --no-build-isolation transformer_engine[jax]

# For both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Or install directly from the GitHub repository:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

Specify frameworks during GitHub installation:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

Install the latest stable version with conda from conda-forge:

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch

# JAX integration (coming soon)
```

#### Source Installation

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

### Environment Variables

Customize the build process with these environment variables:

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   **MAX_JOBS**: Limit parallel build jobs (default varies)
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports both FlashAttention-2 and FlashAttention-3 in PyTorch. FlashAttention-3 is prioritized if both are present.

Verify FlashAttention version using:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Note:** FlashAttention-2 compilation can be resource-intensive.  Consider setting `MAX_JOBS=1` to avoid out-of-memory errors.

## Troubleshooting

**Common Issues and Solutions:**

1.  **ABI Compatibility Issues:**
    *   **Symptoms:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI. Rebuild PyTorch if necessary.
2.  **Missing Headers or Libraries:**
    *   **Symptoms:** CMake errors about missing headers (e.g., `cudnn.h`).
    *   **Solution:** Install missing development packages or set environment variables:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```
        If CMake can't find a C++ compiler, set the `CXX` environment variable.
3.  **Build Resource Issues:**
    *   **Symptoms:** Compilation hangs, system freezes, or OOM errors.
    *   **Solution:** Limit parallel builds:

        ```bash
        MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...
        ```
4.  **Verbose Build Logging:**
    *   For detailed build logs:

        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

The padding mask definition in PyTorch has been changed for consistency.  `True` now means masking out a position, aligning with the other frameworks.

## FP8 Convergence

FP8 has been extensively tested and shows **no significant difference** in convergence compared to BF16.

[List of models tested for convergence](https://github.com/NVIDIA/TransformerEngine#fp8-convergence)

## Integrations

Transformer Engine integrates with popular LLM frameworks:

*   DeepSpeed
*   Hugging Face Accelerate
*   Lightning
*   MosaicML Composer
*   NVIDIA JAX Toolbox
*   NVIDIA Megatron-LM
*   NVIDIA NeMo Framework
*   Amazon SageMaker Model Parallel Library
*   Levanter
*   GPT-NeoX
*   Hugging Face Nanotron - Coming soon!
*   Colossal-AI - Coming soon!
*   PeriFlow - Coming soon!

## Contributing

We welcome contributions! See the `<CONTRIBUTING.rst>`_ guide for instructions.

## Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Videos

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72457/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)

## Previous News

*   [List of previous news](https://github.com/NVIDIA/TransformerEngine#previous-news)