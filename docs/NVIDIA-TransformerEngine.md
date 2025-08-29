# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer models on NVIDIA GPUs with optimized kernels and FP8 support for faster training and inference.** ([Original Repo](https://github.com/NVIDIA/TransformerEngine))

## Key Features

*   **FP8 Support:**  Utilize 8-bit floating-point precision for significant performance gains on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Kernels:** Leverage highly optimized building blocks and fused kernels tailored for Transformer architectures.
*   **Framework Agnostic C++ API:**  Integrate with your existing deep learning libraries to enable FP8 support.
*   **Ease of Use:**  Simplified mixed-precision training with user-friendly Python modules.
*   **Broad Compatibility:**  Support for all precisions (FP16, BF16) on NVIDIA Ampere and later GPU architectures.

## Quickstart

Transformer Engine provides APIs that integrate with popular Large Language Model (LLM) libraries. It provides a Python API consisting of modules to easily build a Transformer layer as well as a framework-agnostic library in C++ including structs and kernels needed for FP8 support. Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly simplifying mixed precision training for users.

### PyTorch Example

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

### JAX (Flax) Example

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
*   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

Leverage pre-built Docker images for the easiest setup, available on the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3  # Example: PyTorch container
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3    # Example: JAX container
```

#### pip Installation

```bash
pip install --no-build-isolation transformer_engine[pytorch]    # PyTorch
pip install --no-build-isolation transformer_engine[jax]       # JAX
pip install --no-build-isolation transformer_engine[pytorch,jax] # Both
```

Or directly from GitHub:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable # Specify frameworks
```

#### conda Installation

```bash
conda install -c conda-forge transformer-engine-torch # PyTorch (from conda-forge)
# JAX integration (coming soon)
```

#### Source Installation

See the installation guide for details: [Installation from Source](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process with these environment variables:

*   **CUDA_PATH:** Path to CUDA installation
*   **CUDNN_PATH:** Path to cuDNN installation
*   **CXX:** Path to C++ compiler
*   **NVTE_FRAMEWORK:**  Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   **MAX_JOBS:** Limit parallel build jobs (default varies by system)
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch for improved performance. FlashAttention-3 was added in release v1.11 and is prioritized over FlashAttention-2 when both are present in the environment.

You can verify which FlashAttention version is being used by setting these environment variables:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

## Troubleshooting

### Common Issues and Solutions

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source if necessary.

2.  **Missing Headers or Libraries:** Install missing development packages or set environment variables (CUDA_PATH, CUDNN_PATH, etc.).  Set CXX if the C++ compiler can't be found.

3.  **Build Resource Issues:** Limit parallel builds: `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...`

4.  **Verbose Build Logging:**  For detailed logs: `cd transformer_engine; pip install -v -v -v --no-build-isolation .`

## Breaking Changes

### v1.7: Padding mask definition for PyTorch

The padding mask has changed from `True` meaning inclusion of the corresponding position in attention to exclusion of that position in our PyTorch implementation. Since v1.7, all attention mask types follow the same definition where `True` means masking out the corresponding position and `False` means including that position in attention calculation.

## FP8 Convergence

FP8 has been extensively tested and shows no significant difference compared to BF16 in training loss curves, with accuracy validated on downstream LLM tasks.

*   **Model Convergence Data:**  See the table in the original README for model details and framework comparisons.

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
*   Hugging Face Nanotron - Coming soon!
*   Colossal-AI - Coming soon!
*   PeriFlow - Coming soon!

## Contributing

Contribute to Transformer Engine by following the guidelines in the `<CONTRIBUTING.rst>`_ guide.

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

*   [List of recent news items from the original README]