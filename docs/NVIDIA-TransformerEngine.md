# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer model training and inference with NVIDIA Transformer Engine, unlocking significant performance gains and reduced memory usage.  [See the original repo](https://github.com/NVIDIA/TransformerEngine).**

## Key Features

*   **FP8 Precision:** Enables faster training and inference on Hopper, Ada, and Blackwell GPUs with 8-bit floating-point (FP8) precision.
*   **Optimized Modules:** Provides easy-to-use modules for building Transformer layers with FP8 support and optimizations like fused kernels.
*   **Framework Agnostic:** Includes a framework-agnostic C++ API for integrating FP8 support with other deep learning libraries.
*   **Mixed Precision Support:**  Supports FP16, BF16, and FP8 across NVIDIA Ampere, Ada, Hopper and Blackwell GPU architectures.
*   **Ease of Use:**  Simplifies mixed-precision training with built-in scaling factor management.
*   **Integration with Leading Frameworks:** Seamlessly integrates with popular LLM frameworks, including PyTorch, JAX, DeepSpeed, and Hugging Face Accelerate.

## Latest News

*   **[03/2025]** `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/>`_
*   **[03/2025]** `Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking <https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/>`_
*   **[02/2025]** `Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2 <https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/>`_
*   **[02/2025]** `NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance <https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/>`_
*   **[01/2025]** `Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud <https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/>`_

## What is Transformer Engine?

Transformer Engine (TE) is a powerful library designed to accelerate Transformer models on NVIDIA GPUs. It leverages FP8 precision on supported GPUs (Hopper, Ada, and Blackwell) to enhance performance while reducing memory usage in both training and inference.  TE provides optimized building blocks tailored for popular Transformer architectures and a mixed-precision-like API.  It also offers a framework-agnostic C++ API.

As Transformer models grow in size, training and inference become increasingly demanding. TE addresses this challenge by offering APIs that integrate with common Large Language Model (LLM) libraries. This provides a Python API with modules for constructing Transformer layers and a C++ library for FP8 support.

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

Flax

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
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended
*   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

Use pre-built Docker images for a streamlined setup from the `NVIDIA GPU Cloud (NGC) Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`.

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

Where 25.04 (corresponding to April 2025 release) is the container version.

**Benefits of using NGC containers:**

*   All dependencies pre-installed with compatible versions and optimized configurations
*   NGC PyTorch 23.08+ containers include FlashAttention-2

#### pip Installation

**Prerequisites:**

*   A compatible C++ compiler
*   CUDA Toolkit with cuDNN and NVCC (NVIDIA CUDA Compiler) installed

```bash
# For PyTorch integration
pip install --no-build-isolation transformer_engine[pytorch]

# For JAX integration
pip install --no-build-isolation transformer_engine[jax]

# For both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Install directly from the GitHub repository:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

Specify frameworks using the environment variable:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch

# JAX integration (coming soon)
```

#### Source Installation

`See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source>`_

### Environment Variables

*   **CUDA_PATH**: Path to CUDA installation
*   **CUDNN_PATH**: Path to cuDNN installation
*   **CXX**: Path to C++ compiler
*   **NVTE_FRAMEWORK**: Comma-separated list of frameworks to build for (e.g., ``pytorch,jax``)
*   **MAX_JOBS**: Limit number of parallel build jobs (default varies by system)
*   **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports both FlashAttention-2 and FlashAttention-3 in PyTorch. FlashAttention-3 is prioritized when present.

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

It is a known issue that FlashAttention-2 compilation is resource-intensive, which may lead to out of memory errors. Please try setting **MAX_JOBS=1** in the environment to circumvent the issue.

## Troubleshooting

### Common Issues and Solutions:

1.  **ABI Compatibility Issues:**
    *   **Symptoms:** `ImportError` with undefined symbols when importing `transformer_engine`
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source with matching ABI.
    *   **Context:** This is particularly common with pip-installed PyTorch outside of containers.

2.  **Missing Headers or Libraries:**
    *   **Symptoms:** CMake errors about missing headers (``cudnn.h``, ``cublas_v2.h``, ``filesystem``, etc.)
    *   **Solution:** Install missing development packages or set environment variables:
        ```bash
        export CUDA_PATH=/path/to/cuda
        export CUDNN_PATH=/path/to/cudnn
        ```
        Set the ``CXX`` environment variable if CMake can't find a C++ compiler.

3.  **Build Resource Issues:**
    *   **Symptoms:** Compilation hangs, system freezes, or out-of-memory errors
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

### v1.7: Padding mask definition for PyTorch

In an effort to unify the definition and usage of the attention mask across all three frameworks in Transformer Engine, the padding mask has changed from `True` meaning inclusion of the corresponding position in attention to exclusion of that position in our PyTorch implementation. Since v1.7, all attention mask types follow the same definition where `True` means masking out the corresponding position and `False` means including that position in attention calculation.

An example of this change is,

```bash
# for a batch of 3 sequences where `a`s, `b`s and `c`s are the useful tokens
# and `0`s are the padding tokens,
[a, a, a, 0, 0,
 b, b, 0, 0, 0,
 c, c, c, c, 0]
# the padding mask for this batch before v1.7 is,
[ True,  True,  True, False, False,
  True,  True, False, False, False,
  True,  True,  True,  True, False]
# and for v1.7 onwards it should be,
[False, False, False,  True,  True,
 False, False,  True,  True,  True,
 False, False, False, False,  True]
```

## FP8 Convergence

FP8 has been tested extensively across different model architectures and configurations with **no significant difference** between FP8 and BF16 training loss curves.

+------------+------------------+---------------------------------------------------------------------------------------------------------+
| Model      | Framework        | Source                                                                                                  |
+============+==================+=========================================================================================================+
| T5-770M    |  JAX/T5x         | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance|
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| MPT-1.3B   |  Mosaic Composer | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                              |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-5B     |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-5B     |  NeMo Framework  | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| LLama2-7B  |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| T5-11B     |  JAX/T5x         | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| MPT-13B    |  Mosaic Composer | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8         |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-22B    |  NeMo Framework  | Available on request                                                                                    |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| LLama2-70B |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
+------------+------------------+---------------------------------------------------------------------------------------------------------+
| GPT-175B   |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
+------------+------------------+---------------------------------------------------------------------------------------------------------+

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
*   Hugging Face Nanotron - *Coming soon!*
*   Colossal-AI - *Coming soon!*
*   PeriFlow - *Coming soon!*

## Contributing

See the `<CONTRIBUTING.rst>`_ guide for contribution guidelines.

## Papers

*   `Attention original paper <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_
*   `Megatron-LM tensor parallel <https://arxiv.org/pdf/1909.08053.pdf>`_
*   `Megatron-LM sequence parallel <https://arxiv.org/pdf/2205.05198.pdf>`_
*   `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_

## Videos

*   `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>`_
*   `Blackwell Numerics for AI | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/>`_
*   `Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025 <https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK>`_
*   `From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/>`_
*   `What's New in Transformer Engine and FP8 Training | GTC 2024 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>`_
*   `FP8 Training with Transformer Engine | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393>`_
*   `FP8 for Deep Learning | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/>`_
*   `Inside the Hopper Architecture | GTC 2022 <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/>`_

## Previous News

*   ... (List of previous news items)