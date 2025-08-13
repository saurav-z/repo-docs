# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Unlock faster training and inference for Transformer models with NVIDIA Transformer Engine, leveraging FP8 precision for superior performance and memory efficiency.** ([Original Repo](https://github.com/NVIDIA/TransformerEngine))

## Key Features

*   **FP8 Support:** Accelerate Transformer models on Hopper, Ada, and Blackwell GPUs using 8-bit floating point (FP8) precision.
*   **Optimized Modules:** Easily build Transformer layers with specialized modules and fused kernels for enhanced performance.
*   **Mixed Precision:** Seamlessly integrate with existing code through a mixed-precision-like API for all precisions (FP16, BF16) and FP8.
*   **Framework Agnostic C++ API:** Integrate with other deep learning libraries for FP8 support in Transformers.
*   **Broad Compatibility:** Supports NVIDIA Ampere, Hopper, Ada, and Blackwell GPU architectures.

## Getting Started

### Quickstart

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

For a comprehensive tutorial, refer to the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

### Installation

Choose your preferred method:

**Docker (Recommended):**

*   Leverage pre-configured Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
*   Example: `docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3` (replace with your desired version).

**pip:**

1.  **Prerequisites:** Compatible C++ compiler, CUDA Toolkit, cuDNN, and NVCC.
2.  Install the latest stable version:

    ```bash
    # For PyTorch integration
    pip install --no-build-isolation transformer_engine[pytorch]
    # For JAX integration
    pip install --no-build-isolation transformer_engine[jax]
    # For both frameworks
    pip install --no-build-isolation transformer_engine[pytorch,jax]
    ```

    Or install directly from GitHub:
    ```bash
    pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
    ```
    Use `NVTE_FRAMEWORK=pytorch,jax` for specifying frameworks during GitHub installation.

**conda:**

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch
```

**Source Installation:**  Refer to the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended
*   **Source Build:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **FP8:** Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Troubleshooting

**Common Issues:**

1.  **ABI Compatibility:** Resolve ``ImportError`` by ensuring PyTorch and Transformer Engine use the same C++ ABI settings.  Rebuild PyTorch if needed.
2.  **Missing Headers/Libraries:** Install development packages and set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`).
3.  **Build Resource Issues:** Limit parallel builds with `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1`
4.  **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .` within the `transformer_engine` directory for detailed logs.

### Compiling with FlashAttention

Enable FlashAttention-2 or FlashAttention-3 (preferred, if present). Verify which FlashAttention version is being used with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py`.
To address potential RAM issues, especially during FlashAttention-2 compilation, try setting `MAX_JOBS=1`.

## Breaking Changes

**v1.7: Padding Mask Definition (PyTorch):**  Padding masks now follow the same definition across all frameworks: `True` indicates masking (exclusion), and `False` indicates inclusion.

## FP8 Convergence

Extensive testing reveals **no significant difference** between FP8 and BF16 training loss curves across various models.  FP8 also maintains accuracy in downstream LLM tasks.

### Examples of Models Tested

*   T5-770M (JAX/T5x)
*   MPT-1.3B (Mosaic Composer)
*   GPT-5B (JAX/Paxml)
*   GPT-5B (NeMo Framework)
*   LLama2-7B (Alibaba Pai)
*   T5-11B (JAX/T5x)
*   MPT-13B (Mosaic Composer)
*   GPT-22B (NeMo Framework)
*   LLama2-70B (Alibaba Pai)
*   GPT-175B (JAX/Paxml)

## Integrations

Transformer Engine seamlessly integrates with:

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

Contribute to Transformer Engine! Review the `<CONTRIBUTING.rst>` guide for detailed instructions.

## Additional Resources

### Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

### Videos

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)

### Previous News

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