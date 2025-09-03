# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Transformer Engine empowers faster and more efficient training and inference of transformer models on NVIDIA GPUs.**

[View the original repository](https://github.com/NVIDIA/TransformerEngine)

## Key Features

*   **FP8 Support:** Accelerates training and inference using 8-bit floating-point (FP8) precision on Hopper, Ada, and Blackwell GPUs, leading to significant performance gains and reduced memory usage.
*   **Optimized Building Blocks:** Provides a collection of highly optimized modules for popular Transformer architectures.
*   **Ease of Use:** Offers a user-friendly API with framework-specific modules for seamless integration with existing PyTorch and JAX code.
*   **Mixed Precision Support:** Includes a mixed-precision-like API and a framework-agnostic C++ API that simplifies the implementation of FP8 and other precision formats (FP16, BF16) for Transformers.
*   **Performance Optimizations:** Leverages fused kernels and other optimizations for enhanced performance.
*   **Broad Compatibility:** Supports NVIDIA Ampere and later GPU architectures.

## Quickstart

Get started with Transformer Engine using the example below:

```python
# PyTorch Example
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

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended

### Installation Methods

**1. Docker (Recommended)**

The easiest way to get started with Transformer Engine is via Docker containers from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
# Example: PyTorch container
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3

# Example: JAX container
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3
```

**2. pip Installation**

```bash
# PyTorch
pip install --no-build-isolation transformer_engine[pytorch]
# JAX
pip install --no-build-isolation transformer_engine[jax]
# Both
pip install --no-build-isolation transformer_engine[pytorch,jax]

# Install from GitHub (stable branch)
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Specify frameworks during GitHub install
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch
```

### Source Installation

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

## Troubleshooting

### Common Issues and Solutions:

**1. ABI Compatibility Issues:**

*   **Symptoms:** `ImportError` with undefined symbols when importing `transformer_engine`.
*   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source if necessary.

**2. Missing Headers or Libraries:**

*   **Symptoms:** CMake errors about missing headers (e.g., `cudnn.h`, `cublas_v2.h`).
*   **Solution:** Install missing development packages and/or set environment variables:

```bash
export CUDA_PATH=/path/to/cuda
export CUDNN_PATH=/path/to/cudnn
```

**3. Build Resource Issues:**

*   **Symptoms:** Compilation hangs, system freezes, or out-of-memory errors.
*   **Solution:** Limit parallel builds:

```bash
MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...
```

**4. Verbose Build Logging:**

*   For detailed build logs:

```bash
cd transformer_engine
pip install -v -v -v --no-build-isolation .
```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

The definition of the padding mask in PyTorch was changed from `True` meaning inclusion of the corresponding position in attention to exclusion of that position. Now all attention mask types follow the same definition.

## FP8 Convergence

Extensive testing across various models and configurations confirms that FP8 training exhibits **no significant difference** in loss curves compared to BF16.

| Model          | Framework        | Source                                                                                                                                                                |
| :------------- | :--------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T5-770M        | JAX/T5x          | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance                                                              |
| MPT-1.3B       | Mosaic Composer  | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                                                                                          |
| GPT-5B         | JAX/Paxml        | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results                                                                             |
| GPT-5B         | NeMo Framework   | Available on request                                                                                                                                                  |
| LLama2-7B      | Alibaba Pai      | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                                                                                     |
| T5-11B         | JAX/T5x          | Available on request                                                                                                                                                  |
| MPT-13B        | Mosaic Composer  | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8                                                                      |
| GPT-22B        | NeMo Framework   | Available on request                                                                                                                                                  |
| LLama2-70B     | Alibaba Pai      | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                                                                                     |
| GPT-175B       | JAX/Paxml        | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results                                                                             |

## Integrations

Transformer Engine seamlessly integrates with popular LLM frameworks:

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

Contributions are welcome! See the `<CONTRIBUTING.rst>`_ guide for details.

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