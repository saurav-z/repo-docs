# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Supercharge your Transformer model performance with NVIDIA Transformer Engine, unlocking faster training and inference through optimized kernels and FP8 support.** [Learn more at the original repository](https://github.com/NVIDIA/TransformerEngine).

## Key Features

*   **FP8 Precision Support:** Leverage 8-bit floating point (FP8) precision on Hopper, Ada, and Blackwell GPUs for improved performance and lower memory utilization.
*   **Optimized Kernels:** Benefit from highly optimized building blocks and fused kernels for Transformer architectures.
*   **Simplified Mixed Precision:**  Easily integrate FP8 training with framework-specific code using an automatic mixed precision-like API.
*   **Framework Agnostic C++ API:** Integrate with other deep learning libraries to enable FP8 support.
*   **Broad Compatibility:**  Supports all precisions (FP16, BF16) on NVIDIA Ampere and later GPU architectures.

## Quickstart

[PyTorch and JAX examples are available in the original README.](https://github.com/NVIDIA/TransformerEngine)

For a more comprehensive tutorial, check out our `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

## Installation

### System Requirements
* **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
* **OS:** Linux (official), WSL2 (limited support)
* **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
* **cuDNN:** 9.3+
* **Compiler:** GCC 9+ or Clang 10+ with C++17 support
* **Python:** 3.12 recommended
* **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

*   **Docker (Recommended):** Utilize pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
    ```
    (Replace `25.04` with the desired container version).

*   **pip:**

    ```bash
    # For PyTorch integration
    pip install --no-build-isolation transformer_engine[pytorch]
    
    # For JAX integration
    pip install --no-build-isolation transformer_engine[jax]
    
    # For both frameworks
    pip install --no-build-isolation transformer_engine[pytorch,jax]
    ```

    or install directly from the GitHub repository:

    ```bash
    pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
    ```

*   **conda:**

    ```bash
    # For PyTorch integration
    conda install -c conda-forge transformer-engine-torch
    
    # JAX integration (coming soon)
    ```

*   **Source:** See the installation guide for details on [Installation from Source](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process with these environment variables:

*   **CUDA_PATH:** Path to CUDA installation
*   **CUDNN_PATH:** Path to cuDNN installation
*   **CXX:** Path to C++ compiler
*   **NVTE_FRAMEWORK:**  Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   **MAX_JOBS:** Limit parallel build jobs (default varies)
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch, prioritized if both are present. Verify the version with:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Troubleshooting**

*   **ABI Compatibility Issues:** If you encounter ``ImportError``, ensure PyTorch and Transformer Engine are built with the same C++ ABI setting.
*   **Missing Headers or Libraries:** Install development packages or set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`).
*   **Build Resource Issues:** Limit parallel builds with `MAX_JOBS=1` if compilation fails.
*   **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .` for detailed logs.

### Breaking Changes

*   **v1.7:** Padding mask definition for PyTorch - now `True` masks out the corresponding position.

## FP8 Convergence

Extensive testing shows **no significant difference** between FP8 and BF16 training loss curves. FP8 accuracy has also been validated on downstream LLM tasks. See the original README for a list of tested models.

## Integrations

Transformer Engine integrates with popular LLM frameworks, including:

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

Contribute to Transformer Engine by following the guidelines in the `<CONTRIBUTING.rst>`_ guide.

## Papers

*   Attention original paper
*   Megatron-LM tensor parallel
*   Megatron-LM sequence parallel
*   FP8 Formats for Deep Learning

## Videos

*   Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025
*   Blackwell Numerics for AI | GTC 2025
*   Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025
*   From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025
*   What's New in Transformer Engine and FP8 Training | GTC 2024
*   FP8 Training with Transformer Engine | GTC 2023
*   FP8 for Deep Learning | GTC 2023
*   Inside the Hopper Architecture | GTC 2022

## Previous News

*   [11/2024] Developing a 172B LLM with Strong Japanese Capabilities Using NVIDIA Megatron-LM
*   [11/2024] How FP8 boosts LLM training by 18% on Amazon SageMaker P5 instances
*   [11/2024] Efficiently train models with large sequence lengths using Amazon SageMaker model parallel
*   [09/2024] Reducing AI large model training costs by 30% requires just a single line of code from FP8 mixed precision training upgrades
*   [05/2024] Accelerating Transformers with NVIDIA cuDNN 9
*   [03/2024] Turbocharged Training: Optimizing the Databricks Mosaic AI stack with FP8
*   [03/2024] FP8 Training Support in SageMaker Model Parallelism Library
*   [12/2023] New NVIDIA NeMo Framework Features and NVIDIA H200
*   [11/2023] Inflection-2: The Next Step Up
*   [11/2023] Unleashing The Power Of Transformers With NVIDIA Transformer Engine
*   [11/2023] Accelerating PyTorch Training Workloads with FP8
*   [09/2023] Transformer Engine added to AWS DL Container for PyTorch Training
*   [06/2023] Breaking MLPerf Training Records with NVIDIA H100 GPUs
*   [04/2023] Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1)