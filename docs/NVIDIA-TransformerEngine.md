# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer models with NVIDIA Transformer Engine, unlocking faster training and inference using FP8 precision, available on Hopper, Ada, and Blackwell GPUs. ([View on GitHub](https://github.com/NVIDIA/TransformerEngine))**

Transformer Engine (TE) is a powerful library designed to accelerate Transformer models on NVIDIA GPUs, offering significant performance gains and reduced memory usage through the use of 8-bit floating-point (FP8) precision, along with other optimizations. This allows for faster training and inference of models like BERT, GPT, and T5, which are becoming increasingly complex.

**Key Features:**

*   **FP8 Precision Support:** Enables faster training and inference using FP8 on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Provides easy-to-use modules for building Transformer layers with FP8 support.
*   **Framework Integration:** Seamless integration with popular deep learning frameworks like PyTorch and JAX.
*   **Performance Optimizations:** Includes fused kernels and other optimizations for improved efficiency.
*   **Mixed-Precision Support:** Supports mixed-precision training (FP16, BF16) on NVIDIA Ampere and later GPU architectures.

## Quickstart

Get started with Transformer Engine with a PyTorch example:

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

For a more comprehensive tutorial, check out our `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

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

#### Docker (Recommended)

Leverage pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for the easiest setup:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3  # Replace 25.04 with the desired version
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

#### pip Installation

*   **PyTorch:** `pip install --no-build-isolation transformer_engine[pytorch]`
*   **JAX:** `pip install --no-build-isolation transformer_engine[jax]`
*   **Both:** `pip install --no-build-isolation transformer_engine[pytorch,jax]`
*   **From GitHub:** `pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`
    *   Specify frameworks during GitHub install:  `NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`

#### conda Installation

*   **PyTorch:**  `conda install -c conda-forge transformer-engine-torch`
*   **JAX (Coming Soon):** [Not yet available]

#### Source Installation

Refer to the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

### Environment Variables

Customize your build with these environment variables:

*   `CUDA_PATH`: Path to CUDA installation
*   `CUDNN_PATH`: Path to cuDNN installation
*   `CXX`: Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated frameworks (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit parallel build jobs
*   `NVTE_BUILD_THREADS_PER_JOB`: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch for performance gains. FlashAttention-3 is prioritized if both are present. Monitor the FlashAttention version used:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Troubleshooting Note:** FlashAttention-2 compilation can be resource-intensive. If you encounter out-of-memory errors, try `MAX_JOBS=1`.

## Troubleshooting

### Common Issues and Solutions:

1.  **ABI Compatibility Issues:**

    *   **Symptoms:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI settings. Rebuild PyTorch from source if necessary.
2.  **Missing Headers or Libraries:**

    *   **Symptoms:** CMake errors about missing headers.
    *   **Solution:** Install missing development packages or set environment variables:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```
        Set `CXX` if CMake can't find a C++ compiler.
3.  **Build Resource Issues:**

    *   **Symptoms:** Compilation hangs, system freezes, or out-of-memory errors.
    *   **Solution:** Limit parallel builds with `MAX_JOBS=1` and `NVTE_BUILD_THREADS_PER_JOB=1`.
4.  **Verbose Build Logging:**

    *   For detailed logs:
        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

The padding mask definition in PyTorch has been updated to align with other frameworks, where `True` means masking out a position.

## FP8 Convergence

Extensive testing demonstrates **no significant difference** in loss curves between FP8 and BF16 training. FP8 has also been validated for accuracy on downstream LLM tasks.

| Model         | Framework       | Source                                                                                                  |
| :------------ | :-------------- | :------------------------------------------------------------------------------------------------------ |
| T5-770M       | JAX/T5x        | [Convergence Details](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance) |
| MPT-1.3B      | Mosaic Composer | [CoreWeave Blog](https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1)                           |
| GPT-5B        | JAX/Paxml       | [H100 Results](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results)           |
| GPT-5B        | NeMo Framework  | Available on request                                                                                    |
| LLama2-7B     | Alibaba Pai     | [WeChat Post](https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ)                                         |
| T5-11B        | JAX/T5x        | Available on request                                                                                    |
| MPT-13B       | Mosaic Composer | [Databricks Blog](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)  |
| GPT-22B       | NeMo Framework  | Available on request                                                                                    |
| LLama2-70B    | Alibaba Pai     | [WeChat Post](https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ)                                         |
| GPT-175B      | JAX/Paxml       | [H100 Results](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results)           |

## Integrations

Transformer Engine is integrated with:

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

*   [Attention Original Paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
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

*(List of recent news items - summarized)*