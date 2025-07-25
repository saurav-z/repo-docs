<!--
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.
-->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Transformer Engine: Accelerate Transformer Models with FP8 on NVIDIA GPUs

**Transformer Engine (TE) enables faster and more efficient training and inference of Transformer models on NVIDIA GPUs through optimized kernels and FP8 precision support, achieving significant performance gains.**  [See the original repo](https://github.com/NVIDIA/TransformerEngine).

## Key Features

*   **FP8 Support:** Accelerate Transformer models using 8-bit floating-point (FP8) precision on NVIDIA Hopper, Ada, and Blackwell GPUs, and other precisions (FP16, BF16) on Ampere and later.
*   **Optimized Kernels:** Leverage highly optimized building blocks and fused kernels for popular Transformer architectures.
*   **Framework Integration:** Seamlessly integrate with PyTorch and JAX, and C++ APIs for broader library compatibility.
*   **Ease of Use:** Utilize easy-to-use modules for building Transformer layers with FP8 support and an autocasting API.
*   **Performance Boost:** Achieve improved performance with lower memory utilization in both training and inference.

## Latest News

*   **[03/2025]** [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   **[03/2025]** [Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/)
    <br>
    ![Comparison of FP8 versus BF16 training, as seen in NVIDIA DGX Cloud Benchmarking Performance Explorer](docs/examples/comparison-fp8-bf16-training-nvidia-dgx-cloud-benchmarking-performance-explorer.jpg)
*   **[02/2025]** [Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2](https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/)
*   **[02/2025]** [NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance](https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/)
*   **[01/2025]** [Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)

## Previous News

[See Previous News](#previous-news)

## What is Transformer Engine?

Transformer Engine (TE) is a library designed to significantly accelerate Transformer models on NVIDIA GPUs. It provides a collection of highly optimized building blocks for popular Transformer architectures and an automatic mixed-precision-like API. A framework agnostic C++ API is also included for integrating with other deep learning libraries. TE supports 8-bit floating point (FP8) precision on Hopper, Ada, and Blackwell GPUs, resulting in better performance and reduced memory usage for both training and inference.

### Highlights

*   Easy-to-use modules for building Transformer layers with FP8 support.
*   Optimizations (e.g. fused kernels) for Transformer models.
*   Support for FP8 on NVIDIA Hopper, Ada, and Blackwell GPUs.
*   Support for optimizations across all precisions (FP16, BF16) on NVIDIA Ampere GPU architecture generations and later.

## Examples

### PyTorch

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

### JAX

#### Flax

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

For a more comprehensive tutorial, check out the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

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

The quickest way to get started with Transformer Engine is by using Docker images on
[NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

For example to use the NGC PyTorch container interactively:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
```

For example to use the NGC JAX container interactively:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

Where 25.04 (corresponding to April 2025 release) is the container version.

**Benefits of using NGC containers:**

*   All dependencies pre-installed with compatible versions and optimized configurations
*   NGC PyTorch 23.08+ containers include FlashAttention-2

#### pip Installation

**Prerequisites for pip installation:**

*   A compatible C++ compiler
*   CUDA Toolkit with cuDNN and NVCC (NVIDIA CUDA Compiler) installed

To install the latest stable version with pip:

```bash
# For PyTorch integration
pip install --no-build-isolation transformer_engine[pytorch]

# For JAX integration
pip install --no-build-isolation transformer_engine[jax]

# For both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Alternatively, install directly from the GitHub repository:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

When installing from GitHub, you can explicitly specify frameworks using the environment variable:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

To install the latest stable version with conda from conda-forge:

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch

# JAX integration (coming soon)
```

#### Source Installation

[See the installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source)

### Environment Variables

These environment variables can be set before installation to customize the build process:

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Comma-separated list of frameworks to build for (e.g., ``pytorch,jax``)
*   **MAX_JOBS**: Limit number of parallel build jobs (default varies by system)
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports both FlashAttention-2 and FlashAttention-3 in PyTorch for improved performance. FlashAttention-3 was added in release v1.11 and is prioritized over FlashAttention-2 when both are present in the environment.

You can verify which FlashAttention version is being used by setting these environment variables:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

It is a known issue that FlashAttention-2 compilation is resource-intensive and requires a large amount of RAM (see [bug](https://github.com/Dao-AILab/flash-attention/issues/358)), which may lead to out of memory errors during the installation of Transformer Engine. Please try setting **MAX_JOBS=1** in the environment to circumvent the issue.

## Troubleshooting

**Common Issues and Solutions:**

1.  **ABI Compatibility Issues:**
    *   **Symptoms:** ``ImportError`` with undefined symbols when importing transformer_engine
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source with matching ABI.
    *   **Context:** If you're using PyTorch built with a different C++ ABI than your system's default, you may encounter these undefined symbol errors. This is particularly common with pip-installed PyTorch outside of containers.

2.  **Missing Headers or Libraries:**
    *   **Symptoms:** CMake errors about missing headers (``cudnn.h``, ``cublas_v2.h``, ``filesystem``, etc.)
    *   **Solution:** Install missing development packages or set environment variables to point to correct locations:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```
    *   If CMake can't find a C++ compiler, set the ``CXX`` environment variable.
    *   Ensure all paths are correctly set before installation.

3.  **Build Resource Issues:**
    *   **Symptoms:** Compilation hangs, system freezes, or out-of-memory errors
    *   **Solution:** Limit parallel builds:

        ```bash
        MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...
        ```

4.  **Verbose Build Logging:**
    *   For detailed build logs to help diagnose issues:

        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding mask definition for PyTorch

In an effort to unify the definition and usage of the attention mask across all three frameworks in Transformer Engine, the padding mask has changed from `True` meaning inclusion of the corresponding position in attention to exclusion of that position in our PyTorch implementation. Since v1.7, all attention mask types follow the same definition where `True` means masking out the corresponding position and `False` means including that position in attention calculation.

An example of this change is,

```bash
# for a batch of 3 sequences where `a`s, `b`s and `c`s are the useful tokens
# and `0`s are the padding tokens,
[a, a, a, 0, 0,
 b, b, 0, 0, 0,
 c, c, c, c, 0]
# the padding mask for this batch before v1.7 is,
[ True,  True,  True, False, False,
  True,  True, False, False, False,
  True,  True,  True,  True, False]
# and for v1.7 onwards it should be,
[False, False, False,  True,  True,
 False, False,  True,  True,  True,
 False, False, False, False,  True]
```

## FP8 Convergence

FP8 has been tested extensively across different model architectures and configurations and we found **no significant difference** between FP8 and BF16 training loss curves. FP8 has also been validated for accuracy on downstream LLM tasks (e.g. LAMBADA and WikiText). Below are examples of models tested for convergence across different frameworks.

| Model        | Framework        | Source                                                                                                                                                                                             |
| :----------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T5-770M      | JAX/T5x          | [https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance) |
| MPT-1.3B     | Mosaic Composer  | [https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1](https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1)                                                                             |
| GPT-5B       | JAX/Paxml        | [https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results)              |
| GPT-5B       | NeMo Framework   | Available on request                                                                                                                                                                             |
| LLama2-7B    | Alibaba Pai      | [https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ](https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ)                                                                                             |
| T5-11B       | JAX/T5x          | Available on request                                                                                                                                                                             |
| MPT-13B      | Mosaic Composer  | [https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)   |
| GPT-22B      | NeMo Framework   | Available on request                                                                                                                                                                             |
| LLama2-70B   | Alibaba Pai      | [https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ](https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ)                                                                                             |
| GPT-175B     | JAX/Paxml        | [https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results)              |

## Integrations

Transformer Engine is integrated with popular LLM frameworks, including:

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

## Contributing

We welcome contributions! Please follow the guidelines in the  `<CONTRIBUTING.rst>`_ guide.

## Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Videos

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)