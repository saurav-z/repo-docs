# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

Transformer Engine empowers faster and more efficient training and inference for Transformer models, especially with 8-bit floating-point (FP8) precision, offering significant performance gains and reduced memory usage.  Access the original repo [here](https://github.com/NVIDIA/TransformerEngine).

**Key Features:**

*   **FP8 Precision:** Accelerate Transformer models on Hopper, Ada, and Blackwell GPUs using FP8, providing performance boosts with minimal accuracy loss.
*   **Optimized Building Blocks:**  Leverage highly optimized modules for popular Transformer architectures.
*   **Automatic Mixed Precision-like API:** Easily integrate FP8 into your existing PyTorch and JAX code.
*   **Framework-Agnostic C++ API:** Integrate FP8 support with other deep learning libraries.
*   **Optimizations for All Precisions:** Support for FP16, BF16, and FP8 across various NVIDIA GPU architectures (Ampere and later).
*   **Integrated with Popular Frameworks:**  Works seamlessly with PyTorch, JAX, and other LLM libraries.

## Quickstart

Get started with a quick example to see how Transformer Engine can be implemented:

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
**JAX (Flax)**
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

For a comprehensive tutorial, explore our `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

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

The easiest way to get started is with Docker images from the `NVIDIA GPU Cloud (NGC) Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_.

For PyTorch:
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
```
For JAX:
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3
```
Where 25.08 (corresponding to August 2025 release) is the container version.

**Benefits of using NGC containers:**

*   All dependencies pre-installed with compatible versions and optimized configurations
*   NGC PyTorch 23.08+ containers include FlashAttention-2

**2. pip Installation**

**Prerequisites for pip installation:**

*   A compatible C++ compiler
*   CUDA Toolkit with cuDNN and NVCC (NVIDIA CUDA Compiler) installed

To install the latest stable version:

```bash
# For PyTorch integration
pip install --no-build-isolation transformer_engine[pytorch]
    
# For JAX integration
pip install --no-build-isolation transformer_engine[jax]
    
# For both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Install directly from GitHub:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

Specify frameworks with the environment variable:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch
    
# JAX integration (coming soon)
```

**4. Source Installation**

`See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source>`_

### Environment Variables

Customize the build process with these environment variables:

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Frameworks to build for (e.g., `pytorch,jax`)
*   **MAX_JOBS**: Limit parallel build jobs (default varies)
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch.  FlashAttention-3 is prioritized when both are present.

Verify FlashAttention version:
```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Note:** FlashAttention-2 compilation can be resource-intensive.  Address potential out-of-memory errors during installation by setting `MAX_JOBS=1`.

## Troubleshooting

**Common Issues and Solutions:**

1.  **ABI Compatibility:**  ``ImportError`` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI. Rebuild PyTorch if needed.
2.  **Missing Headers/Libraries:** CMake errors about missing headers (e.g., `cudnn.h`, `cublas_v2.h`).
    *   **Solution:** Install development packages or set environment variables like `CUDA_PATH` and `CUDNN_PATH`.  Set `CXX` if a C++ compiler isn't found.
3.  **Build Resource Issues:** Compilation hangs or out-of-memory errors.
    *   **Solution:** Limit parallel builds: `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...`
4.  **Verbose Build Logging:** For detailed build logs:
    *   **Solution:** `cd transformer_engine; pip install -v -v -v --no-build-isolation .`

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

Since v1.7, the padding mask in PyTorch has been redefined.  `True` in the mask now *excludes* the corresponding position from attention, aligning with the standard across all frameworks.

## FP8 Convergence

Extensive testing shows **no significant difference** between FP8 and BF16 training loss curves. FP8 has been validated for accuracy on downstream LLM tasks (e.g. LAMBADA and WikiText).

[Table of tested models and frameworks - Refer to original README for the table]

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

Contributions are welcome!  Refer to the `<CONTRIBUTING.rst>`_ guide.

## Papers

*   Attention original paper
*   Megatron-LM tensor parallel
*   Megatron-LM sequence parallel
*   FP8 Formats for Deep Learning

## Videos

[List of videos - Refer to original README for the list]

## Previous News

[List of news - Refer to original README for the list]