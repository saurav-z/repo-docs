# Transformer Engine: Accelerate Transformer Models with FP8 Precision 🚀

[View the original repository](https://github.com/NVIDIA/TransformerEngine)

Transformer Engine (TE) is a powerful library designed to **significantly accelerate the training and inference of Transformer models on NVIDIA GPUs**, particularly by leveraging the efficiency of 8-bit floating point (FP8) precision.

**Key Features:**

*   **FP8 Acceleration:** Utilize FP8 precision on NVIDIA Hopper, Ada, and Blackwell GPUs for faster performance and reduced memory usage.
*   **Optimized Modules:** Easily build Transformer layers with optimized, fused kernels for maximum performance.
*   **Framework Agnostic C++ API:** Integrate FP8 support into other deep learning libraries.
*   **Mixed Precision Support:** Optimized for FP16, BF16, and FP8 precision across NVIDIA architectures.
*   **Simplified API:** Python API with modules to streamline the creation of Transformer layers with FP8 support.

**Quick Links:**

*   [Quickstart](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb)
*   [Installation](#installation)
*   [User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
*   [Examples](https://github.com/NVIDIA/TransformerEngine/tree/main/examples)

## Latest News

*   **[03/2025]** [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   **[03/2025]** [Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/)
    ![FP8 vs. BF16 Training Performance](docs/examples/comparison-fp8-bf16-training-nvidia-dgx-cloud-benchmarking-performance-explorer.jpg)
*   **[02/2025]** [Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2](https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/)
*   **[02/2025]** [NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance](https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/)
*   **[01/2025]** [Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)

## What is Transformer Engine?

Transformer Engine empowers you to build and train cutting-edge Transformer models with unparalleled speed and efficiency. It provides optimized building blocks for popular Transformer architectures and a user-friendly API that integrates seamlessly with your existing deep learning code. Transformer Engine streamlines the use of mixed-precision training, particularly FP8, offering significant performance gains with minimal accuracy trade-offs.

## Highlights

*   Easy-to-use modules for building Transformer layers with FP8 support
*   Optimizations (e.g. fused kernels) for Transformer models
*   Support for FP8 on NVIDIA Hopper, Ada, and Blackwell GPUs
*   Support for optimizations across all precisions (FP16, BF16) on NVIDIA Ampere GPU architecture generations and later

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

For a comprehensive tutorial, explore the `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`.

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

The easiest way to get started is by using Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

Example:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
```

Replace `25.08` with the desired container version.

**Benefits of NGC containers:**

*   Pre-installed dependencies with compatible versions.
*   Optimized configurations.
*   NGC PyTorch 23.08+ containers include FlashAttention-2.

#### pip Installation

**Prerequisites:**

*   A compatible C++ compiler
*   CUDA Toolkit with cuDNN and NVCC installed

Install the latest stable version with pip:

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

Specify frameworks using the environment variable when installing from GitHub:

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

Customize the build process using these environment variables before installation:

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Comma-separated list of frameworks to build for (e.g., `pytorch,jax`)
*   **MAX_JOBS**: Limit number of parallel build jobs (default varies by system)
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch. FlashAttention-3 (v1.11+) is prioritized when both are present.

Verify FlashAttention version:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Troubleshooting**

[See the troubleshooting section in original README]

## Breaking Changes

[See the breaking changes section in original README]

## FP8 Convergence

FP8 has been extensively tested and shows **no significant difference** in training loss compared to BF16. It has also been validated for accuracy on downstream LLM tasks.

[See the table in the original README]

## Integrations

Transformer Engine is integrated with popular LLM frameworks:

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

We welcome contributions! Follow the guidelines in the `<CONTRIBUTING.rst>` guide.

## Papers

[See the papers section in the original README]

## Videos

[See the videos section in the original README]

## Previous News

[See the previous news section in the original README]