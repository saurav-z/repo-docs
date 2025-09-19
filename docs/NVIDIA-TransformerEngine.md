# Transformer Engine: Accelerate Transformer Models with FP8 Precision

Transformer Engine is a powerful library designed to significantly accelerate Transformer models on NVIDIA GPUs, empowering faster training and inference with reduced memory usage.  [Explore the project on GitHub!](https://github.com/NVIDIA/TransformerEngine)

## Key Features

*   **FP8 Support:** Leverage 8-bit floating point (FP8) precision on NVIDIA Hopper, Ada, and Blackwell GPUs for optimized performance.
*   **Optimized Building Blocks:** Access a collection of highly optimized components for popular Transformer architectures.
*   **Mixed Precision API:** Utilize an automatic mixed-precision API for seamless integration with your existing framework code.
*   **Framework Agnostic C++ API:** Integrate the C++ API with other deep learning libraries to enable FP8 support for Transformers.
*   **Performance Boosts:** Experience significant speedups and lower memory utilization in both training and inference.
*   **Ease of Use:** Benefit from easy-to-use modules for building Transformer layers.
*   **Broad Compatibility:** Support for all precisions (FP16, BF16, FP8) on various NVIDIA GPU architectures.

## Quickstart

Get started with Transformer Engine quickly using the provided code examples.

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

### JAX (Flax)

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

For a more comprehensive tutorial, check out our [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

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

Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for easy setup.

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3 # PyTorch
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3 # JAX
```

**2. pip Installation**

*   **PyTorch:** `pip install --no-build-isolation transformer_engine[pytorch]`
*   **JAX:** `pip install --no-build-isolation transformer_engine[jax]`
*   **Both:** `pip install --no-build-isolation transformer_engine[pytorch,jax]`

Alternatively, install from the GitHub repository:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
# Specify frameworks
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
conda install -c conda-forge transformer-engine-torch # PyTorch (JAX coming soon)
```

**4. Source Installation**

Follow the instructions in the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process using these environment variables:

*   `CUDA_PATH`: Path to CUDA installation
*   `CUDNN_PATH`: Path to cuDNN installation
*   `CXX`: Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated list of frameworks to build for (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit number of parallel build jobs (default varies)
*   `NVTE_BUILD_THREADS_PER_JOB`: Control threads per build job

### Compiling with FlashAttention

Supports FlashAttention-2 and FlashAttention-3.  Verify with:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```
To circumvent out of memory issues during installation, set `MAX_JOBS=1`.

### Troubleshooting

**Common Issues and Solutions:**

1.  **ABI Compatibility Issues:**
    *   **Symptoms:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI settings. Rebuild PyTorch from source with matching ABI.
2.  **Missing Headers or Libraries:**
    *   **Symptoms:** CMake errors about missing headers (e.g., `cudnn.h`, `cublas_v2.h`).
    *   **Solution:** Install missing development packages or set environment variables:

        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```

    *   Set the `CXX` environment variable if CMake can't find a C++ compiler.
3.  **Build Resource Issues:**
    *   **Symptoms:** Compilation hangs or out-of-memory errors.
    *   **Solution:** Limit parallel builds: `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...`
4.  **Verbose Build Logging:**
    *   For detailed build logs:

        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

Since v1.7, the padding mask in PyTorch uses the same definition as the other frameworks: `True` means masking out the corresponding position.

## FP8 Convergence

Extensive testing shows **no significant difference** in loss curves between FP8 and BF16 training.

**Models Tested:**  (See the original README for a complete table.)

## Integrations

Transformer Engine seamlessly integrates with leading LLM frameworks: (See the original README for a complete list.)

## Contributing

Contribute to Transformer Engine by following the guidelines in the `<CONTRIBUTING.rst>` guide.

## Papers

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Videos

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72457/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK>)

## Previous News

(See the original README for a complete list.)