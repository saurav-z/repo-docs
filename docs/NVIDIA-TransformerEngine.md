<!-- Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- See LICENSE for license information. -->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Transformer Engine is a powerful library that unlocks significant performance and memory efficiency for training and inference of Transformer models on NVIDIA GPUs, particularly with FP8 precision.**  For more information, see the [original repository](https://github.com/NVIDIA/TransformerEngine).

## Key Features

*   **FP8 Support:** Optimized for FP8 precision on Hopper, Ada, and Blackwell GPUs, enabling faster training and inference with lower memory usage.
*   **Performance Boost:** Provides significant speedups for Transformer models compared to FP32 and FP16 training.
*   **Ease of Use:** Offers user-friendly modules for building Transformer layers, along with a framework-agnostic C++ API.
*   **Optimized Kernels:** Includes highly optimized kernels and fused operations tailored for Transformer architectures.
*   **Broad Framework Compatibility:** Integrates with popular frameworks like PyTorch, JAX, and others, with support for FP16, BF16, and FP8.

## Quickstart

### PyTorch Example

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

### JAX/Flax Example

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
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended
*   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

**1. Docker (Recommended)**

Leverage pre-built, optimized Docker images on the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
# PyTorch
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3

# JAX
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

**2. pip Installation**

Ensure you have a compatible C++ compiler and the CUDA Toolkit installed.

```bash
# PyTorch integration
pip install --no-build-isolation transformer_engine[pytorch]

# JAX integration
pip install --no-build-isolation transformer_engine[jax]

# Both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]

# From GitHub (specify frameworks)
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
# PyTorch integration
conda install -c conda-forge transformer-engine-torch
# JAX integration (coming soon)
```

**4. Source Installation**

Follow the detailed instructions in the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process with these environment variables:

*   `CUDA_PATH`: Path to CUDA installation
*   `CUDNN_PATH`: Path to cuDNN installation
*   `CXX`: Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit parallel build jobs (default varies)
*   `NVTE_BUILD_THREADS_PER_JOB`: Threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 for enhanced performance in PyTorch (FlashAttention-3 is prioritized if both are present).  To verify which version is used:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

Note: FlashAttention-2 compilation can be resource-intensive; consider setting `MAX_JOBS=1` if encountering out-of-memory errors.

### Troubleshooting

#### Common Issues and Solutions

1.  **ABI Compatibility Issues:**

    *   **Symptoms:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI.  Rebuild PyTorch if necessary.  This is especially critical if using a pip-installed PyTorch outside of a container.

2.  **Missing Headers or Libraries:**

    *   **Symptoms:** CMake errors about missing headers (`cudnn.h`, `cublas_v2.h`, `filesystem`, etc.).
    *   **Solution:** Install missing development packages or set environment variables:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```
    *   If CMake can't find the compiler, set the `CXX` environment variable.
    *   Ensure all paths are correctly set before installation.

3.  **Build Resource Issues:**

    *   **Symptoms:** Compilation hangs, system freezes, or out-of-memory errors.
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

### Breaking Changes

#### v1.7: Padding Mask Definition for PyTorch

In v1.7, the padding mask definition for PyTorch was changed to be consistent with other frameworks.  `True` in the mask now indicates that a position should be *masked out* (excluded from attention).

### FP8 Convergence

Extensive testing across various models and configurations has shown **no significant difference** in loss curves between FP8 and BF16 training. Accuracy on downstream LLM tasks has also been validated.

*   [Model Convergence Table (T5-770M, MPT-1.3B, GPT-5B, LLama2-7B, etc.)](See original README for the table - too large to repeat here.)

### Integrations

Transformer Engine is integrated with the following LLM frameworks:

*   [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/runtime/half_precision/test_fp8.py)
*   [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/low_precision_training#configuring-transformersengine)
*   [Lightning](https://github.com/Lightning-AI/lightning/issues/17172)
*   [MosaicML Composer](https://github.com/mosaicml/composer/releases/tag/v0.13.1)
*   [NVIDIA JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox)
*   [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
*   [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo-Megatron-Launcher)
*   [Amazon SageMaker Model Parallel Library](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-tensor-parallelism.html)
*   [Levanter](https://github.com/stanford-crfm/levanter)
*   [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
*   [Hugging Face Nanotron](https://github.com/huggingface/nanotron) - Coming soon!
*   [Colossal-AI](https://github.com/hpcaitech/ColossalAI) - Coming soon!
*   [PeriFlow](https://github.com/friendliai/periflow-python-sdk) - Coming soon!

### Contributing

We welcome contributions!  See the `<CONTRIBUTING.rst>`_ guide for details.

### Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

### Videos

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)

### Previous News

*   [Blog Posts and Announcements](See original README)