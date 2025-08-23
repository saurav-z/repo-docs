# Transformer Engine: Accelerating Transformer Models on NVIDIA GPUs

**Supercharge your Transformer model performance with NVIDIA Transformer Engine, enabling faster training and inference through FP8 precision and optimized kernels.  [View the original repository](https://github.com/NVIDIA/TransformerEngine)**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Key Features

*   **FP8 Support:**  Leverage 8-bit floating point (FP8) precision for faster training and reduced memory usage on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Easily build Transformer layers with optimized kernels and fused operations.
*   **Framework Agnostic C++ API:** Integrate FP8 support into existing deep learning libraries.
*   **Mixed-Precision API:**  Seamlessly integrates with your existing PyTorch and JAX code for mixed-precision training.
*   **Optimizations:**  Benefit from performance optimizations across various precisions (FP16, BF16).

## Quickstart

Get started with Transformer Engine quickly:

*   [Quickstart](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb)
*   [User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)

## Installation

Transformer Engine offers several installation methods to suit your needs:

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell)
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended

### Installation Methods

#### Docker (Recommended)

The fastest way to get started, using pre-built Docker images:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3
```

Replace `25.08` with the desired container version (e.g., August 2025 release).

#### pip Installation

Install the latest stable version with pip:

```bash
# For PyTorch
pip install --no-build-isolation transformer_engine[pytorch]
# For JAX
pip install --no-build-isolation transformer_engine[jax]
# For both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Or, install directly from the GitHub repository:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

Specify frameworks during GitHub installation:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

Install the latest stable version with conda from conda-forge:

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch
# JAX integration (coming soon)
```

#### Source Installation

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

### Environment Variables

Customize the build process with these environment variables:

*   `CUDA_PATH`:  Path to CUDA installation
*   `CUDNN_PATH`: Path to cuDNN installation
*   `CXX`:  Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated frameworks (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit parallel build jobs (varies by system)
*   `NVTE_BUILD_THREADS_PER_JOB`: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 (v1.11+), with FlashAttention-3 prioritized if both are present.

Verify the FlashAttention version being used:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Note:**  FlashAttention-2 compilation can be resource-intensive. Address potential out-of-memory errors during installation by setting `MAX_JOBS=1`.

### Examples

#### PyTorch

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

#### JAX

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

## Troubleshooting

### Common Issues and Solutions

1.  **ABI Compatibility Issues:**

    *   **Symptom:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine use the same C++ ABI. Rebuild PyTorch from source if necessary.

2.  **Missing Headers or Libraries:**

    *   **Symptom:** CMake errors about missing headers (`cudnn.h`, `cublas_v2.h`, `filesystem`, etc.)
    *   **Solution:** Install missing development packages or set environment variables:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```

    *   If CMake can't find the C++ compiler, set the `CXX` environment variable.
    *   Verify paths are correctly set before installation.

3.  **Build Resource Issues:**

    *   **Symptom:** Compilation hangs, freezes, or out-of-memory errors.
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

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

The padding mask definition in PyTorch changed to align with other frameworks: `True` now indicates *masking out* (excluding) a position in the attention calculation.

## FP8 Convergence

Extensive testing shows **no significant difference** between FP8 and BF16 training loss curves.  FP8 accuracy has been validated on downstream LLM tasks.

## Integrations

Transformer Engine is integrated with popular LLM frameworks, including:

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

Contribute to Transformer Engine by following the guidelines in the `<CONTRIBUTING.rst>` guide.

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