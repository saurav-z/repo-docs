# Transformer Engine: Accelerate Transformer Models with FP8 Precision

Transformer Engine is a powerful library from NVIDIA designed to supercharge the performance of Transformer models on NVIDIA GPUs, offering significant speedups and reduced memory usage, particularly with 8-bit floating-point (FP8) precision.  [Explore the original repository](https://github.com/NVIDIA/TransformerEngine).

**Key Features:**

*   **FP8 Support:** Accelerate training and inference on Hopper, Ada, and Blackwell GPUs using FP8 precision.
*   **Optimized Modules:** Includes easy-to-use modules for building Transformer layers.
*   **Fused Kernels:** Provides optimized kernels for enhanced performance of Transformer models.
*   **Mixed Precision:** Supports FP16, BF16, and FP8 optimizations across various NVIDIA GPU architectures (Ampere and later).
*   **Framework Agnostic C++ API:** Integrate FP8 support into other deep learning libraries.
*   **Integration with popular LLM frameworks:** Seamless integration with Deepspeed, Hugging Face Accelerate, Pytorch Lightning, and more.

## Quickstart

Get up and running with Transformer Engine using the quickstart guides for PyTorch and JAX.

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

For a more detailed walkthrough, check out the `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

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

The easiest way to get started. Use NGC containers, which have all dependencies pre-installed:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3  # PyTorch
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3 # JAX
```

Replace `25.04` with the desired container version.

**2. pip Installation**

Ensure a compatible C++ compiler and CUDA Toolkit are installed.

```bash
pip install --no-build-isolation transformer_engine[pytorch]  # PyTorch
pip install --no-build-isolation transformer_engine[jax] # JAX
pip install --no-build-isolation transformer_engine[pytorch,jax] # Both
```

or, install from GitHub:
```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
conda install -c conda-forge transformer-engine-torch # PyTorch - JAX coming soon
```
**4. Source Installation**

[See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source>`_]

### Environment Variables

Customize the build process using these environment variables:

*   **CUDA_PATH:** Path to CUDA installation
*   **CUDNN_PATH:** Path to cuDNN installation
*   **CXX:** Path to C++ compiler
*   **NVTE_FRAMEWORK:** Frameworks to build for (e.g., ``pytorch,jax``)
*   **MAX_JOBS:** Limit parallel build jobs
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per build job

## FP8 Convergence

Extensive testing across different model architectures shows **no significant difference** in training loss curves between FP8 and BF16.  FP8 has also demonstrated accuracy on downstream LLM tasks.

## Troubleshooting

### Common Issues and Solutions:
1. **ABI Compatibility Issues:** `ImportError` with undefined symbols. Ensure PyTorch and Transformer Engine use the same C++ ABI. Rebuild PyTorch if necessary.
2. **Missing Headers or Libraries:** CMake errors. Install missing development packages or set environment variables (CUDA_PATH, CUDNN_PATH, CXX).
3. **Build Resource Issues:** Compilation hangs/freezes/OOM. Limit parallel builds with `MAX_JOBS=1`.
4. **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .`

## Breaking Changes

v1.7: The padding mask definition was changed to `True` meaning masking out the corresponding position and `False` meaning including that position in attention calculation

## Additional Resources

*   [User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
*   [Examples](https://github.com/NVIDIA/TransformerEngine/tree/main/examples)
*   [Release Notes](https://docs.nvidia.com/deeplearning/transformer-engine/documentation-archive.html)
*   [Contribute](https://github.com/NVIDIA/TransformerEngine/blob/main/CONTRIBUTING.rst)
*   [Videos](https://www.nvidia.com/en-us/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [Papers](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)