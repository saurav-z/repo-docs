# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Supercharge your Transformer models with optimized performance and reduced memory usage using NVIDIA's Transformer Engine.**

[View the original repository on GitHub](https://github.com/NVIDIA/TransformerEngine)

## Key Features

*   **FP8 Support:** Leverage 8-bit floating point (FP8) precision for faster training and inference on NVIDIA Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Utilize pre-built, highly optimized modules for building Transformer layers.
*   **Mixed Precision:** Achieve significant speedups with minimal accuracy loss by using mixed-precision training across all supported precisions (FP16, BF16).
*   **Framework Agnostic C++ API:** Integrate with other deep learning libraries to enable FP8 support for Transformers.
*   **Easy Integration:** Seamlessly integrate with popular LLM frameworks like PyTorch and JAX.

## Quickstart

Get up and running with Transformer Engine using these links:

*   [Quickstart](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb)
*   [Examples](https://github.com/NVIDIA/TransformerEngine/tree/main/examples)
*   [User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)

## Latest News

*   [03/2025] `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/>`_
*   [03/2025] `Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking <https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/>`_

    .. image:: docs/examples/comparison-fp8-bf16-training-nvidia-dgx-cloud-benchmarking-performance-explorer.jpg
      :width: 600
      :alt: Comparison of FP8 versus BF16 training, as seen in NVIDIA DGX Cloud Benchmarking Performance Explorer

*   [02/2025] `Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2 <https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/>`_
*   [02/2025] `NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance <https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/>`_
*   [01/2025] `Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud <https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/>`_

[Previous News](#previous-news)

## What is Transformer Engine?

Transformer Engine (TE) is a library designed to accelerate Transformer models on NVIDIA GPUs. It provides optimized building blocks and an easy-to-use API for implementing FP8 precision on compatible GPUs, improving performance and reducing memory usage during training and inference.

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell)
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended

### Installation Methods

**1. Docker (Recommended)**

The easiest way to start with Transformer Engine is using pre-built Docker images on the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

**2. pip Installation**

*   **For PyTorch:** `pip install --no-build-isolation transformer_engine[pytorch]`
*   **For JAX:** `pip install --no-build-isolation transformer_engine[jax]`
*   **For Both:** `pip install --no-build-isolation transformer_engine[pytorch,jax]`

**3. conda Installation**

```bash
conda install -c conda-forge transformer-engine-torch
```

**4. Source Installation**

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

## Environment Variables

Customize the build process with these environment variables:

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   **MAX_JOBS**: Limit parallel build jobs
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3. Set these environment variables for verbose build logging:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

## Troubleshooting

**Common Issues and Solutions:**

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting.
2.  **Missing Headers or Libraries:** Install missing development packages or set environment variables:

    ```bash
    export CUDA_PATH=/path/to/cuda
    export CUDNN_PATH=/path/to/cudnn
    ```

3.  **Build Resource Issues:** Limit parallel builds with `MAX_JOBS=1`.
4.  **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .` for detailed logs.

## Breaking Changes

*   **v1.7: Padding Mask Definition for PyTorch:**  Padding mask now uses the same definition across all frameworks. `True` means masking out the corresponding position.

## FP8 Convergence

FP8 has been rigorously tested and shows **no significant difference** in loss curves compared to BF16.

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
*   Hugging Face Nanotron (Coming soon!)
*   Colossal-AI (Coming soon!)
*   PeriFlow (Coming soon!)

## Contributing

We welcome contributions!  Follow the guidelines in the `<CONTRIBUTING.rst>`_ guide.

## Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Videos

*   `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>`__
*   `Blackwell Numerics for AI | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/>`_
*   `Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025 <https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK>`_
*   `From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/>`_
*   `What's New in Transformer Engine and FP8 Training | GTC 2024 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>`_
*   `FP8 Training with Transformer Engine | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393>`_
*   `FP8 for Deep Learning | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/>`_
*   `Inside the Hopper Architecture | GTC 2022 <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/>`_