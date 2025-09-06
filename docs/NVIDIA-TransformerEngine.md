# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Transformer Engine (TE) is a powerful library designed to supercharge the performance of Transformer models on NVIDIA GPUs, offering significant speedups and memory savings through innovative techniques like FP8 precision.**  [Explore the original repository](https://github.com/NVIDIA/TransformerEngine)

## Key Features

*   **FP8 Support:** Utilize 8-bit floating point (FP8) precision for faster training and inference on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Building Blocks:** Leverage highly optimized components for popular Transformer architectures.
*   **Simplified Mixed Precision:** Integrate seamlessly with your existing framework code using an automatic mixed-precision API.
*   **Framework Agnostic C++ API:**  Enable FP8 support for Transformers in any deep learning library.
*   **FP16/BF16 Optimizations:** Benefit from performance enhancements across all supported precisions (FP16, BF16) on Ampere and later architectures.
*   **Ease of Use:** Simplify mixed-precision training with easy-to-use modules for building Transformer layers.

## Quickstart

Get started with Transformer Engine using these example code snippets:

**PyTorch**

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

**JAX/Flax**

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

For a comprehensive tutorial, check out the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

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

**1. Docker (Recommended)**

Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for a streamlined setup.

```bash
# PyTorch
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3

# JAX
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

# From GitHub (latest stable)
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Specify frameworks during GitHub installation
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
# PyTorch
conda install -c conda-forge transformer-engine-torch

# JAX (coming soon)
```

**4. Source Installation**

[See the installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source)

### Environment Variables

Customize your build with these environment variables:

*   **CUDA_PATH:** Path to CUDA installation
*   **CUDNN_PATH:** Path to cuDNN installation
*   **CXX:** Path to C++ compiler
*   **NVTE_FRAMEWORK:** Comma-separated frameworks (e.g., `pytorch,jax`)
*   **MAX_JOBS:** Limit parallel build jobs
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 for improved performance in PyTorch.  Verify the version being used:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```
*   Setting **MAX_JOBS=1** may resolve potential out-of-memory errors during FlashAttention-2 compilation

## Troubleshooting

**Common Issues and Solutions:**

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting.
2.  **Missing Headers or Libraries:** Install missing development packages or set environment variables: `CUDA_PATH`, `CUDNN_PATH`.
3.  **Build Resource Issues:** Limit parallel builds with `MAX_JOBS=1`.
4.  **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .` inside the `transformer_engine` directory.

## Breaking Changes

**v1.7: Padding Mask Definition for PyTorch**

The padding mask definition has changed in v1.7, where `True` now indicates masking out a position (exclusion from attention), aligning with the definition across all frameworks.

## FP8 Convergence

Extensive testing across various model architectures has shown **no significant difference** between FP8 and BF16 training loss curves.  FP8 has also been validated for downstream LLM tasks.

[See the table in the original README for specific model examples.]

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
*   Hugging Face Nanotron - *Coming soon!*
*   Colossal-AI - *Coming soon!*
*   PeriFlow - *Coming soon!*

## Contributing

We welcome contributions!  Follow the guidelines in the `<CONTRIBUTING.rst>` guide.

## Papers and Videos

[Links to relevant research papers and videos are available in the original README.]