# Transformer Engine: Accelerate Transformer Models with FP8 Precision 🚀

**Supercharge your Transformer models on NVIDIA GPUs with Transformer Engine, achieving significant performance gains and reduced memory usage through optimized building blocks and FP8 support.  [Explore the original repository](https://github.com/NVIDIA/TransformerEngine).**

## Key Features

*   **FP8 Support:** Leverage 8-bit floating point (FP8) precision for faster training and inference on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Utilize easy-to-use modules and fused kernels specifically designed for Transformer models.
*   **Framework Agnostic C++ API:** Integrate with various deep learning libraries to enable FP8 support.
*   **Mixed Precision API:**  Simplified mixed-precision training with a Python API, internally managing scaling factors.
*   **Performance Boosts:** Experience significant speedups with minimal accuracy loss using mixed-precision techniques like FP8, FP16, and BF16.
*   **Broad Compatibility:** Optimized for NVIDIA Hopper, Ada, and Blackwell GPU architectures (and Ampere for other precisions), and integrates with popular frameworks.

## Quickstart

Get started with the latest features and integrations:
*   `Quickstart <#examples>`_
*   `Installation <#installation>`_
*   `User Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_
*   `Examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_

## Latest News

*   [03/2025] `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/>`_
*   [03/2025] `Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking <https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/>`_

## Benefits

*   **Faster Training:** Achieve significant speedups compared to FP32 training.
*   **Reduced Memory Usage:**  Lower memory footprint, enabling larger models and batch sizes.
*   **Accuracy Preservation:** Minimal accuracy degradation when using lower precision formats.

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

For a more comprehensive tutorial, check out our `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell)
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17
*   **Python:** 3.12 recommended
*   **Source Build:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **FP8:** Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
# PyTorch
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
# JAX
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

#### pip Installation

```bash
# PyTorch
pip install --no-build-isolation transformer_engine[pytorch]
# JAX
pip install --no-build-isolation transformer_engine[jax]
# Both
pip install --no-build-isolation transformer_engine[pytorch,jax]
# From GitHub (Stable)
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
# Specifying frameworks
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

```bash
# PyTorch
conda install -c conda-forge transformer-engine-torch
# JAX (Coming soon)
```

#### Source Installation

`See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source>`_

### Environment Variables

Customize build with these environment variables:

*   **CUDA_PATH:** Path to CUDA installation
*   **CUDNN_PATH:** Path to cuDNN installation
*   **CXX:** Path to C++ compiler
*   **NVTE_FRAMEWORK:** Frameworks to build (e.g., `pytorch,jax`)
*   **MAX_JOBS:** Limit parallel build jobs
*   **NVTE_BUILD_THREADS_PER_JOB:** Threads per build job

### Compiling with FlashAttention

Verify FlashAttention version with:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Troubleshooting:** Consider setting `MAX_JOBS=1` to avoid out-of-memory errors during FlashAttention-2 compilation.

## Troubleshooting

### Common Issues and Solutions:

1.  **ABI Compatibility:**  `ImportError` due to ABI mismatch. Rebuild PyTorch or set C++ ABI.
2.  **Missing Headers/Libraries:** CMake errors. Install missing dependencies or set environment variables (CUDA_PATH, CUDNN_PATH, CXX).
3.  **Build Resource Issues:** Compilation hangs/OOM errors. Limit parallel builds with `MAX_JOBS=1`.
4.  **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .` for detailed logs.

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

Padding masks now use the same definition across all frameworks, where `True` indicates masking out a position.

## FP8 Convergence

FP8 has been rigorously tested and demonstrates **no significant difference** in convergence compared to BF16 across multiple model architectures and configurations.  Accuracy has been validated on downstream tasks.

## Integrations

Transformer Engine integrates seamlessly with popular LLM frameworks, including:

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

Contribute to Transformer Engine!  Follow the guidelines in the `<CONTRIBUTING.rst>`_ guide.

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