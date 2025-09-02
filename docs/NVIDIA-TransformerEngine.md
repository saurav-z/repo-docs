# Transformer Engine: Accelerate Transformer Models with FP8 on NVIDIA GPUs

**Supercharge your Transformer model training and inference with NVIDIA Transformer Engine, achieving higher performance and lower memory utilization using FP8 precision.**  [See the original repository](https://github.com/NVIDIA/TransformerEngine).

*   **FP8 Support:** Leverage 8-bit floating-point precision for significant performance gains and reduced memory usage on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Utilize easy-to-use modules designed for building Transformer layers, with built-in optimizations.
*   **Framework Agnostic C++ API:** Seamlessly integrate FP8 support into existing deep learning libraries with a framework-agnostic C++ API.
*   **Mixed Precision Support:** Benefit from optimizations across various precisions (FP16, BF16) on NVIDIA Ampere and newer GPU architectures.
*   **Broad Integration:**  Integrates with popular LLM frameworks like DeepSpeed, Hugging Face Accelerate, and more.

## Latest News

*   **[03/2025]** [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   **[03/2025]** [Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/)
*   **[02/2025]** [Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2](https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/)
*   **[02/2025]** [NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance](https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/)
*   **[01/2025]** [Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)

## What is Transformer Engine?

Transformer Engine (TE) is a library designed to accelerate Transformer models on NVIDIA GPUs. It provides optimized building blocks and an automatic mixed precision-like API, including support for 8-bit floating point (FP8) precision on Hopper, Ada, and Blackwell GPUs. TE offers better performance and lower memory utilization during both training and inference. It simplifies mixed-precision training by providing APIs that integrate with popular Large Language Model (LLM) libraries and includes a framework-agnostic C++ API.

## Highlights

*   **FP8 Acceleration:**  Achieve significant speedups and reduced memory usage with FP8 support.
*   **Optimized Building Blocks:**  Leverage highly optimized modules for Transformer layers.
*   **Broad GPU Support:** Works with Blackwell, Hopper, Ada, and Ampere GPU architectures.
*   **Simplified Mixed Precision:** Built-in scaling factors and an autocasting API simplify mixed-precision training.

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

**Flax**

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

For a comprehensive tutorial, consult the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended
*   **Source Build:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **FP8:** Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3
```

#### pip Installation

**Prerequisites:** Compatible C++ compiler, CUDA Toolkit with cuDNN, and NVCC.

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

Specify frameworks using the environment variable:

```bash
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

```bash
# For PyTorch integration
conda install -c conda-forge transformer-engine-torch
```
### Source Installation

Follow the detailed [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process with these environment variables:

*   **CUDA_PATH:** CUDA installation path
*   **CUDNN_PATH:** cuDNN installation path
*   **CXX:** C++ compiler path
*   **NVTE_FRAMEWORK:** Frameworks to build for (e.g., `pytorch,jax`)
*   **MAX_JOBS:** Limit parallel build jobs
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per build job

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 for improved performance.  Verify FlashAttention version:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

Set `MAX_JOBS=1` to avoid potential out-of-memory errors during FlashAttention-2 compilation.

### Troubleshooting

#### Common Issues

1.  **ABI Compatibility Issues:**
    *   **Symptom:** `ImportError` with undefined symbols.
    *   **Solution:** Rebuild PyTorch with the same C++ ABI as your system.
2.  **Missing Headers/Libraries:**
    *   **Symptom:** CMake errors about missing headers.
    *   **Solution:** Install development packages or set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`, `CXX`).
3.  **Build Resource Issues:**
    *   **Symptom:** Compilation hangs or out-of-memory errors.
    *   **Solution:** Limit parallel builds using `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1`.
4.  **Verbose Build Logging:**
    *   **Solution:**  Use detailed logs for diagnosing issues: `cd transformer_engine; pip install -v -v -v --no-build-isolation .`

#### Breaking Changes

*   **v1.7:**  Padding mask definition changed in PyTorch (True = mask out).

## FP8 Convergence

Extensive testing across different model architectures demonstrates **no significant difference** in training loss curves between FP8 and BF16. FP8 has also been validated for accuracy on downstream LLM tasks.

| Model       | Framework       | Source                                                                                                  |
| ----------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| T5-770M     | JAX/T5x        | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance |
| MPT-1.3B    | Mosaic Composer | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                              |
| GPT-5B      | JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
| GPT-5B      | NeMo Framework  | Available on request                                                                                    |
| LLama2-7B   | Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
| T5-11B      | JAX/T5x        | Available on request                                                                                    |
| MPT-13B     | Mosaic Composer | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8         |
| GPT-22B     | NeMo Framework  | Available on request                                                                                    |
| LLama2-70B  | Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
| GPT-175B    | JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |

## Integrations

Transformer Engine integrates with leading LLM frameworks:

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