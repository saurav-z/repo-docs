# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Supercharge your Transformer model training and inference with NVIDIA Transformer Engine, achieving faster performance and reduced memory usage!** ([Original Repo](https://github.com/NVIDIA/TransformerEngine))

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Key Features

*   **FP8 Support:** Enables the use of 8-bit floating-point (FP8) precision on Hopper, Ada, and Blackwell GPUs for faster training and inference.
*   **Optimized Modules:** Provides easy-to-use modules for building Transformer layers with FP8 support, including fused kernels for improved performance.
*   **Framework Agnostic C++ API:** Integrates with popular deep learning libraries for FP8 support, offering flexibility across different frameworks.
*   **Mixed Precision:** Supports optimizations across FP16, BF16, and FP8 on NVIDIA Ampere and later GPU architectures.
*   **Seamless Integration:**  Offers a Python API that integrates with popular Large Language Model (LLM) libraries.

## Latest News

*   **[03/2025]**  [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   **[03/2025]**  [Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/)
    ![Comparison of FP8 versus BF16 training, as seen in NVIDIA DGX Cloud Benchmarking Performance Explorer](docs/examples/comparison-fp8-bf16-training-nvidia-dgx-cloud-benchmarking-performance-explorer.jpg)
*   **[02/2025]**  [Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2](https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/)
*   **[02/2025]**  [NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance](https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/)
*   **[01/2025]**  [Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)
[Previous News](#previous-news)

## What is Transformer Engine?

Transformer Engine (TE) is a powerful library designed to accelerate Transformer models on NVIDIA GPUs. It leverages cutting-edge techniques, including 8-bit floating-point (FP8) precision, to significantly improve performance and reduce memory consumption during both training and inference. TE provides optimized building blocks and an easy-to-use API that seamlessly integrates with your existing deep learning framework code.

As Transformer models grow in complexity, memory and compute demands increase. TE addresses this challenge by enabling mixed-precision training, allowing you to combine FP32 with lower precision formats like FP16 and FP8. This approach delivers substantial speedups with minimal impact on accuracy. With Hopper and later GPU architectures, FP8 precision further boosts performance compared to FP16, without sacrificing accuracy.

TE offers Python and C++ APIs for building Transformer layers and incorporating FP8 support. It simplifies mixed-precision training by managing the necessary scaling factors and internal values.

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

For a comprehensive tutorial, check out the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

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

Get started quickly using Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

*   **PyTorch:**

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
    ```

*   **JAX:**

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
    ```

    (Where `25.04` is the container version for the April 2025 release.)

#### pip Installation

**Prerequisites:** Compatible C++ compiler, CUDA Toolkit with cuDNN and NVCC.

*   **Latest Stable Version:**

    ```bash
    # For PyTorch integration
    pip install --no-build-isolation transformer_engine[pytorch]

    # For JAX integration
    pip install --no-build-isolation transformer_engine[jax]

    # For both frameworks
    pip install --no-build-isolation transformer_engine[pytorch,jax]
    ```

*   **From GitHub:**

    ```bash
    pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
    ```

    Specify frameworks using `NVTE_FRAMEWORK`:

    ```bash
    NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
    ```

#### conda Installation

*   **Latest Stable Version (from conda-forge):**

    ```bash
    # For PyTorch integration
    conda install -c conda-forge transformer-engine-torch

    # JAX integration (coming soon)
    ```

#### Source Installation

See the [Installation Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process:

*   `CUDA_PATH`: Path to CUDA
*   `CUDNN_PATH`: Path to cuDNN
*   `CXX`: Path to C++ compiler
*   `NVTE_FRAMEWORK`: Comma-separated list of frameworks (e.g., `pytorch,jax`)
*   `MAX_JOBS`: Limit parallel build jobs (default varies)
*   `NVTE_BUILD_THREADS_PER_JOB`: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 (v1.11 and later prioritizes FlashAttention-3 if both are installed) in PyTorch.  FlashAttention-2 compilation can be resource-intensive; consider setting `MAX_JOBS=1` if you encounter out-of-memory errors.

Verify FlashAttention version:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

## Troubleshooting

### Common Issues and Solutions

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine use the same C++ ABI. Rebuild PyTorch if necessary.
2.  **Missing Headers/Libraries:** Install development packages or set environment variables like `CUDA_PATH` and `CUDNN_PATH`. Set `CXX` if CMake can't find a compiler.
3.  **Build Resource Issues:** Limit parallel builds with `MAX_JOBS=1` and `NVTE_BUILD_THREADS_PER_JOB=1`.
4.  **Verbose Build Logging:**  Use `pip install -v -v -v --no-build-isolation .` within the `transformer_engine` directory.

### Breaking Changes

#### v1.7: Padding Mask Definition (PyTorch)

The padding mask definition in PyTorch has changed to align with other frameworks. In v1.7 and later, `True` in the mask now *excludes* the corresponding position from attention.

## FP8 Convergence

Extensive testing across various models shows **no significant difference** in loss curves between FP8 and BF16 training.  FP8 has also been validated for downstream LLM tasks.

**Model Convergence Examples:**
(Table of tested models is included in original)

## Integrations

Transformer Engine is integrated with these LLM frameworks:

(List of integrations is included in original)

## Contributing

We welcome contributions! Follow the guidelines in the `<CONTRIBUTING.rst>` guide.

## Papers

(List of papers is included in original)

## Videos

(List of videos is included in original)

## Previous News

(List of previous news items is included in original)