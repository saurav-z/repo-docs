# Transformer Engine: Accelerate Transformer Models with FP8 Precision on NVIDIA GPUs

**Supercharge your Transformer model training and inference with NVIDIA Transformer Engine, achieving faster performance and lower memory utilization.**  ([Original Repo](https://github.com/NVIDIA/TransformerEngine))

[Quickstart](https://github.com/NVIDIA/TransformerEngine#examples) | [Installation](https://github.com/NVIDIA/TransformerEngine#installation) | [User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) | [Examples](https://github.com/NVIDIA/TransformerEngine/tree/main/examples) | [FP8 Convergence](https://github.com/NVIDIA/TransformerEngine#fp8-convergence) | [Integrations](https://github.com/NVIDIA/TransformerEngine#integrations) | [Release notes](https://docs.nvidia.com/deeplearning/transformer-engine/documentation-archive.html)

## Key Features

*   **FP8 Precision Support:** Accelerate training and inference on Hopper, Ada, and Blackwell GPUs using 8-bit floating point (FP8) precision.
*   **Optimized Modules:** Utilize easy-to-use modules for building Transformer layers, including fused kernels, for enhanced performance.
*   **Mixed-Precision API:** Seamlessly integrate with your existing framework-specific code using an automatic mixed-precision API.
*   **Framework Agnostic C++ API:**  Integrate with other deep learning libraries to enable FP8 support for Transformers.
*   **Broad Compatibility:** Supports FP16, BF16, and FP8 precision across NVIDIA Ampere, Ada, Hopper, and Blackwell GPU architectures.

## Latest News

*   **[03/2025]** [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   **[03/2025]** [Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/)
    ![FP8 vs BF16 training performance comparison](./docs/examples/comparison-fp8-bf16-training-nvidia-dgx-cloud-benchmarking-performance-explorer.jpg)
*   **[02/2025]** [Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2](https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/)
*   **[02/2025]** [NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance](https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/)
*   **[01/2025]** [Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)

[Previous News](https://github.com/NVIDIA/TransformerEngine#previous-news)

## What is Transformer Engine?

Transformer Engine (TE) is a powerful library designed to significantly accelerate the training and inference of Transformer models on NVIDIA GPUs. It achieves this by leveraging 8-bit floating point (FP8) precision on Hopper, Ada, and Blackwell GPUs. This results in both faster performance and reduced memory consumption. TE offers a comprehensive suite of optimized building blocks tailored for popular Transformer architectures and provides a user-friendly, mixed-precision-like API that seamlessly integrates with your existing framework code.  In addition, the library includes a framework-agnostic C++ API.  This allows you to integrate FP8 support into other deep learning libraries.

Transformer models, such as BERT, GPT, and T5, are becoming increasingly complex, requiring substantial compute and memory resources. While many deep learning frameworks default to FP32, mixed-precision training – combining FP32 with lower precision formats like FP16 and, now, FP8 – can deliver significant speedups with minimal impact on accuracy. The Hopper GPU architecture introduced FP8, which further improves performance compared to FP16, without sacrificing accuracy. Transformer Engine overcomes the existing lack of native FP8 support in major deep learning frameworks by providing APIs that work with popular Large Language Model (LLM) libraries.

### Highlights:

*   **Simplified FP8 Implementation:** Easy-to-use modules streamline the construction of Transformer layers with FP8 capabilities.
*   **Optimized Kernels:** Benefit from optimizations, including fused kernels, tailored specifically for Transformer models.
*   **Broad GPU Support:** Compatible with NVIDIA Hopper, Ada, and Blackwell GPUs for FP8, and all precisions on NVIDIA Ampere and later architectures.

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

Flax

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

For a detailed tutorial, explore the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

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

The easiest way to begin with Transformer Engine is using Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

To use the NGC PyTorch container interactively:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
```

To use the NGC JAX container interactively:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

*Where 25.04 (April 2025 release) is the container version.*

**Benefits of NGC containers:**

*   All dependencies are pre-installed with compatible versions and optimized configurations.
*   NGC PyTorch 23.08+ containers include FlashAttention-2.

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

When installing from GitHub, specify frameworks using the environment variable:

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

These environment variables can be set before installation for customization:

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Comma-separated list of frameworks to build for (e.g., ``pytorch,jax``)
*   **MAX_JOBS**: Limit number of parallel build jobs (default varies by system)
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch for improved performance. FlashAttention-3 was added in release v1.11 and is prioritized over FlashAttention-2 when both are present in the environment.

Verify FlashAttention version using:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

FlashAttention-2 compilation can be resource-intensive. Resolve out-of-memory errors during installation by setting: **MAX_JOBS=1**

### Troubleshooting

#### Common Issues and Solutions

1.  **ABI Compatibility Issues:**

    *   **Symptoms:** `ImportError` with undefined symbols when importing transformer\_engine
    *   **Solution:** Ensure PyTorch and Transformer Engine share the same C++ ABI setting. Rebuild PyTorch from source with matching ABI.
    *   **Context:** These errors arise when PyTorch is built with a different C++ ABI than the system's default, which is common with pip-installed PyTorch outside containers.

2.  **Missing Headers or Libraries:**

    *   **Symptoms:** CMake errors related to missing headers (`cudnn.h`, `cublas_v2.h`, `filesystem`, etc.)
    *   **Solution:** Install development packages or set environment variables:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```

        If CMake cannot find a C++ compiler, set the `CXX` environment variable. Verify all paths before installation.

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

### Breaking Changes

#### v1.7: Padding mask definition for PyTorch

In v1.7, the definition of the padding mask was updated in the PyTorch implementation to align with the other frameworks. Now, `True` indicates masking out (excluding) the corresponding position, and `False` indicates including it.

## FP8 Convergence

FP8 has been extensively tested, showing **no significant difference** between FP8 and BF16 training loss curves across various model architectures and configurations. FP8 accuracy has been validated on downstream LLM tasks.

+------------+------------------+---------------------------------------------------------------------------------------------------------+
| Model      | Framework        | Source                                                                                                  |
+============+==================+=========================================================================================================+
| T5-770M    |  JAX/T5x         | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance|
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| MPT-1.3B   |  Mosaic Composer | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                              |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-5B     |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-5B     |  NeMo Framework  | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| LLama2-7B  |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| T5-11B     |  JAX/T5x         | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| MPT-13B    |  Mosaic Composer | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8         |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-22B    |  NeMo Framework  | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| LLama2-70B |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-175B   |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
+------------+------------------+---------------------------------------------------------------------------------------------------------+

## Integrations

Transformer Engine integrates with leading LLM frameworks:

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

Contributions are welcome!  See the [CONTRIBUTING.rst](https://github.com/NVIDIA/TransformerEngine/blob/main/CONTRIBUTING.rst) guide for details.

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

## Previous News
*   [11/2024] Developing a 172B LLM with Strong Japanese Capabilities Using NVIDIA Megatron-LM
*   [11/2024] How FP8 boosts LLM training by 18% on Amazon SageMaker P5 instances
*   [11/2024] Efficiently train models with large sequence lengths using Amazon SageMaker model parallel
*   [09/2024] Reducing AI large model training costs by 30% requires just a single line of code from FP8 mixed precision training upgrades
*   [05/2024] Accelerating Transformers with NVIDIA cuDNN 9
*   [03/2024] Turbocharged Training: Optimizing the Databricks Mosaic AI stack with FP8
*   [03/2024] FP8 Training Support in SageMaker Model Parallelism Library
*   [12/2023] New NVIDIA NeMo Framework Features and NVIDIA H200
    ![H200 NeMo performance](./docs/examples/H200-NeMo-performance.png)
*   [11/2023] Inflection-2: The Next Step Up
*   [11/2023] Unleashing The Power Of Transformers With NVIDIA Transformer Engine
*   [11/2023] Accelerating PyTorch Training Workloads with FP8
*   [09/2023] Transformer Engine added to AWS DL Container for PyTorch Training
*   [06/2023] Breaking MLPerf Training Records with NVIDIA H100 GPUs
*   [04/2023] Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1)