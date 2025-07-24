# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer models with NVIDIA Transformer Engine, unlocking faster training and inference with FP8 precision on NVIDIA GPUs.**  [View the original repository](https://github.com/NVIDIA/TransformerEngine)

## Key Features

*   **FP8 Support:** Accelerate training and inference on NVIDIA Hopper, Ada, and Blackwell GPUs using 8-bit floating point (FP8) precision.
*   **Optimized Modules:**  Utilize easy-to-use modules for building Transformer layers with optimized kernels for peak performance.
*   **Mixed Precision:** Seamlessly integrate with your existing PyTorch and JAX code for mixed precision training (FP16, BF16, and FP8).
*   **Framework Agnostic C++ API:** Integrate FP8 support into other deep learning libraries.
*   **Performance Boost:** Achieve significant speedups in training and inference with minimal impact on accuracy.
*   **Broad Compatibility:** Supports NVIDIA Ampere, Ada, Hopper, and Blackwell GPU architectures.

## Quickstart

Get started with Transformer Engine using the following examples:

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

For a more detailed walkthrough, explore our  [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended
*   **Source Build:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

Use pre-built Docker images on the  [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
# Example PyTorch container
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3

# Example JAX container
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```
*Replace `25.04` with the desired container version.*

**Benefits of NGC containers:**
*   Pre-installed dependencies with optimized configurations
*   NGC PyTorch 23.08+ containers include FlashAttention-2

#### pip Installation

**Prerequisites:** Compatible C++ compiler and CUDA Toolkit with cuDNN and NVCC.

```bash
# PyTorch
pip install --no-build-isolation transformer_engine[pytorch]

# JAX
pip install --no-build-isolation transformer_engine[jax]

# Both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```
Alternatively, install from the GitHub repository:
```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```
Specify frameworks during GitHub installation:
```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```
#### conda Installation

```bash
# PyTorch
conda install -c conda-forge transformer-engine-torch

# JAX (coming soon)
```

#### Source Installation

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

### Environment Variables

Customize the build process with these environment variables:

*   `CUDA_PATH`: Path to CUDA installation
*   `CUDNN_PATH`: Path to cuDNN installation
*   `CXX`: Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated frameworks (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit parallel build jobs
*   `NVTE_BUILD_THREADS_PER_JOB`: Threads per build job

### Compiling with FlashAttention

Transformer Engine supports both FlashAttention-2 and FlashAttention-3 (v1.11 and later).  FlashAttention-3 is prioritized if both are present.

Verify which FlashAttention version is being used with:
```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```
*   **Note:** FlashAttention-2 compilation can be RAM-intensive.  Address potential out-of-memory errors by setting `MAX_JOBS=1`.

## Troubleshooting

#### Common Issues and Solutions

1.  **ABI Compatibility:**
    *   **Symptom:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source with matching ABI.
2.  **Missing Headers or Libraries:**
    *   **Symptom:** CMake errors about missing headers (e.g., `cudnn.h`, `cublas_v2.h`).
    *   **Solution:** Install missing development packages or set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`). If CMake can't find a C++ compiler, set the `CXX` environment variable.
3.  **Build Resource Issues:**
    *   **Symptom:** Compilation hangs, system freezes, or out-of-memory errors.
    *   **Solution:** Limit parallel builds: `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...`
4.  **Verbose Build Logging:**
    *   Get detailed build logs:
        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

The padding mask definition in PyTorch has changed from `True` representing inclusion to exclusion, aligning with the other frameworks. In the updated definition, `True` signifies masking out a position, while `False` indicates inclusion.

## FP8 Convergence

Extensive testing across various models and configurations has shown **no significant difference** between FP8 and BF16 training loss curves. FP8 accuracy has been validated on downstream LLM tasks.

| Model      | Framework        | Source                                                                                                  |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
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

Transformer Engine is integrated with these popular LLM frameworks:

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

We welcome contributions! See the  `<CONTRIBUTING.rst>`  guide for details.

## Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

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

(See original for complete list of dates and links)