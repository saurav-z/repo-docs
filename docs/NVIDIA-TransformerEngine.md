# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Transformer Engine empowers faster and more efficient training and inference for Transformer models using optimized building blocks and FP8 precision, unlocking significant performance gains.**  [View the original repository](https://github.com/NVIDIA/TransformerEngine)

Key Features:

*   **FP8 Precision Support:** Leverage 8-bit floating point (FP8) for Hopper, Ada, and Blackwell GPUs to reduce memory usage and boost performance.
*   **Optimized Modules:** Utilize pre-built, highly optimized modules for Transformer layers, reducing development time.
*   **Framework Agnostic C++ API:** Integrate FP8 support with your existing deep learning libraries.
*   **Mixed Precision API:**  Easily implement mixed-precision training with an autocasting API.
*   **Performance Optimizations:** Benefit from fused kernels and other optimizations for Transformer models.
*   **Broad Hardware Support:** Compatible with NVIDIA Blackwell, Hopper, Grace Hopper/Blackwell, Ada, and Ampere GPUs.

## Quickstart

Refer to the  `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>` for a comprehensive tutorial.

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

##### Flax

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

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere GPUs
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended
*   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

The easiest way to get started is via NGC containers:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

#### pip Installation

```bash
pip install --no-build-isolation transformer_engine[pytorch] # for PyTorch
pip install --no-build-isolation transformer_engine[jax]     # for JAX
pip install --no-build-isolation transformer_engine[pytorch,jax] # for both
```

Alternatively, install from GitHub:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

Or specify the framework with environment variable:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

```bash
conda install -c conda-forge transformer-engine-torch # for PyTorch
```
(JAX integration coming soon.)

#### Source Installation

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source) for detailed instructions.

### Environment Variables

Customize the build with these environment variables:

*   **CUDA_PATH:** Path to CUDA installation
*   **CUDNN_PATH:** Path to cuDNN installation
*   **CXX:** Path to C++ compiler
*   **NVTE_FRAMEWORK:** Frameworks to build for (e.g., `pytorch,jax`)
*   **MAX_JOBS:** Limit parallel build jobs
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports both FlashAttention-2 and FlashAttention-3 in PyTorch. FlashAttention-3 is prioritized if both are present.  Verify the FlashAttention version with:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

To avoid potential out-of-memory errors during FlashAttention-2 compilation, try setting `MAX_JOBS=1`.

## Troubleshooting

### Common Issues and Solutions:

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI. Rebuild PyTorch from source if needed.
2.  **Missing Headers or Libraries:** Install missing development packages or set environment variables for CUDA/cuDNN:

    ```bash
    export CUDA_PATH=/path/to/cuda
    export CUDNN_PATH=/path/to/cudnn
    ```
    Set the `CXX` environment variable if CMake can't find a C++ compiler.
3.  **Build Resource Issues:** Limit parallel builds to resolve compilation issues:

    ```bash
    MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...
    ```
4.  **Verbose Build Logging:** Get detailed build logs for diagnostics:

    ```bash
    cd transformer_engine
    pip install -v -v -v --no-build-isolation .
    ```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

The padding mask definition for PyTorch has been unified. `True` now means masking out the corresponding position, and `False` means including it.

## FP8 Convergence

Extensive testing shows **no significant difference** between FP8 and BF16 training loss curves.

[Detailed results and model comparisons are available in the original README.](https://github.com/NVIDIA/TransformerEngine)

## Integrations

Transformer Engine integrates with popular LLM frameworks including:

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

Contribute to Transformer Engine following the [CONTRIBUTING.rst](https://github.com/NVIDIA/TransformerEngine/blob/main/CONTRIBUTING.rst) guidelines.

## Papers

*   [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Videos

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)

## Previous News

*   (See original README for recent news)