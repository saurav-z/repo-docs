# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Transformer Engine empowers faster and more memory-efficient training and inference of Transformer models on NVIDIA GPUs.** ([See the original repo](https://github.com/NVIDIA/TransformerEngine))

## Key Features

*   **FP8 Support:** Enables 8-bit floating point (FP8) precision for significant performance gains on NVIDIA Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Provides easy-to-use building blocks for constructing Transformer layers.
*   **Fused Kernels:** Leverages optimized kernels for enhanced performance.
*   **Framework Agnostic:** Offers a C++ API for integration with various deep learning libraries, supporting FP8.
*   **Mixed Precision:** Facilitates mixed-precision training (FP32/FP16/BF16/FP8) for speed and efficiency.
*   **Broad Hardware Support:** Optimized for NVIDIA Blackwell, Hopper, Grace Hopper/Blackwell, Ada, and Ampere GPUs.

## Quickstart

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

*   Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
*   Example (PyTorch): `docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3`
*   Example (JAX): `docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3`

**2. pip Installation**

*   **Prerequisites:** Compatible C++ compiler, CUDA Toolkit with cuDNN and NVCC installed.
*   Install the latest stable version:

    ```bash
    # For PyTorch integration
    pip install --no-build-isolation transformer_engine[pytorch]
    
    # For JAX integration
    pip install --no-build-isolation transformer_engine[jax]
    
    # For both frameworks
    pip install --no-build-isolation transformer_engine[pytorch,jax]
    ```

*   Or, install from GitHub:
    `pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`
    Specify frameworks during GitHub install:
    `NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`

**3. conda Installation**

*   Install the latest stable version from conda-forge:

    ```bash
    # For PyTorch integration
    conda install -c conda-forge transformer-engine-torch
    # JAX integration (coming soon)
    ```

**4. Source Installation**

*   Refer to the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process with these environment variables:

*   `CUDA_PATH`: Path to CUDA installation
*   `CUDNN_PATH`: Path to cuDNN installation
*   `CXX`: Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit parallel build jobs (default varies)
*   `NVTE_BUILD_THREADS_PER_JOB`: Control threads per build job

### Compiling with FlashAttention

*   Transformer Engine supports FlashAttention-2 and FlashAttention-3 (v1.11+). FlashAttention-3 is prioritized if both are present.
*   Debug FlashAttention version:
    `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py`
*   **Troubleshooting**: FlashAttention-2 compilation can be resource-intensive, so try setting `MAX_JOBS=1` if encountering issues.

### Troubleshooting

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. If you're using PyTorch built with a different C++ ABI than your system's default, you may encounter these undefined symbol errors. This is particularly common with pip-installed PyTorch outside of containers.
    *   **Symptom:** `ImportError` with undefined symbols.
    *   **Solution:** Rebuild PyTorch from source with matching ABI.

2.  **Missing Headers or Libraries:** Install missing development packages or set environment variables.
    *   **Symptom:** CMake errors about missing headers (``cudnn.h``, ``cublas_v2.h``, etc.)
    *   **Solution:**
        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```
        If CMake can't find a C++ compiler, set the `CXX` environment variable.

3.  **Build Resource Issues:**
    *   **Symptom:** Compilation hangs, system freezes, or out-of-memory errors
    *   **Solution:** Limit parallel builds: `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...`

4.  **Verbose Build Logging:**
    *   For detailed build logs:
        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

*   **v1.7: Padding Mask Definition for PyTorch**: The definition of the attention mask changed to follow the same definition across all frameworks. `True` means masking out the corresponding position and `False` means including that position in attention calculation.

## FP8 Convergence

FP8 demonstrates comparable convergence to BF16 across various model architectures and configurations, as validated for accuracy on downstream LLM tasks.

| Model       | Framework      | Source                                                                                                                                  |
| :---------- | :------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| T5-770M     | JAX/T5x        | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance                             |
| MPT-1.3B    | Mosaic Composer | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                                                              |
| GPT-5B      | JAX/Paxml      | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results                                             |
| GPT-5B      | NeMo Framework | Available on request                                                                                                                    |
| LLama2-7B   | Alibaba Pai    | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                                                       |
| T5-11B      | JAX/T5x        | Available on request                                                                                                                    |
| MPT-13B     | Mosaic Composer | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8                                           |
| GPT-22B     | NeMo Framework | Available on request                                                                                                                    |
| LLama2-70B  | Alibaba Pai    | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                                                       |
| GPT-175B    | JAX/Paxml      | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results                                             |

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
*   Hugging Face Nanotron (Coming soon!)
*   Colossal-AI (Coming soon!)
*   PeriFlow (Coming soon!)

## Contributing

Contributions are welcome! Please follow the guidelines in the `<CONTRIBUTING.rst>` guide.

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
*   [11/2023] Inflection-2: The Next Step Up
*   [11/2023] Unleashing The Power Of Transformers With NVIDIA Transformer Engine
*   [11/2023] Accelerating PyTorch Training Workloads with FP8
*   [09/2023] Transformer Engine added to AWS DL Container for PyTorch Training
*   [06/2023] Breaking MLPerf Training Records with NVIDIA H100 GPUs
*   [04/2023] Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1)