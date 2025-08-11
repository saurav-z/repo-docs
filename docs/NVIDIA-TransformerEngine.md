# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer model training and inference with NVIDIA Transformer Engine, unlocking significant performance gains and reduced memory usage. [Learn more at the original repository](https://github.com/NVIDIA/TransformerEngine).**

## Key Features

*   **FP8 Support:** Leverage 8-bit floating point (FP8) precision for faster training and inference on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Utilize easy-to-use modules specifically designed for building Transformer layers.
*   **Performance Boosts:** Benefit from optimizations like fused kernels for improved efficiency.
*   **Framework Compatibility:** Supports FP8 and other precision levels (FP16, BF16) across NVIDIA Ampere and later GPU architectures.
*   **Seamless Integration:** Integrates easily with popular LLM libraries and frameworks.

## What is Transformer Engine?

Transformer Engine (TE) is a powerful library meticulously crafted to accelerate the performance of Transformer models on NVIDIA GPUs. Designed to improve both training and inference, TE utilizes cutting-edge techniques, including 8-bit floating point (FP8) precision on supported GPUs (Hopper, Ada, and Blackwell). This precision enables remarkable performance improvements while significantly reducing memory utilization. TE's design provides a collection of optimized building blocks for prevalent Transformer architectures and an automatic mixed precision-like API that is compatible with your existing framework-specific code. For developers seeking to integrate FP8 support into other deep learning libraries, TE offers a framework-agnostic C++ API.

Transformer models are known to be computationally intensive, especially as they become more complex. TE addresses these challenges by providing a more efficient alternative to FP32 training, without compromising accuracy. By integrating FP8 precision, TE offers significant speedups compared to FP16 training, with minimal impact on accuracy.

TE’s APIs seamlessly integrate with leading Large Language Model (LLM) libraries and provides a Python API that provides modules that help build Transformer layers, and a framework-agnostic library in C++ that simplifies mixed precision training.

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

For a detailed tutorial, see the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

## Installation

### System Requirements

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
*   **OS:** Linux (official), WSL2 (limited support)
*   **Software:**
    *   CUDA: 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell)
    *   cuDNN: 9.3+
    *   Compiler: GCC 9+ or Clang 10+
    *   Python: 3.12 recommended
*   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
*   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

### Installation Methods

#### Docker (Recommended)

Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3
```

#### pip Installation

*   **Prerequisites:** Compatible C++ compiler, CUDA Toolkit, and cuDNN.

```bash
pip install --no-build-isolation transformer_engine[pytorch]
pip install --no-build-isolation transformer_engine[jax]
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Alternatively, install from the GitHub repository:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### conda Installation

```bash
conda install -c conda-forge transformer-engine-torch
# JAX integration (coming soon)
```

#### Source Installation

See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Customize the build process using these environment variables:

*   **CUDA_PATH:** CUDA installation path.
*   **CUDNN_PATH:** cuDNN installation path.
*   **CXX:** C++ compiler path.
*   **NVTE_FRAMEWORK:** Frameworks to build for (e.g., `pytorch,jax`).
*   **MAX_JOBS:** Limit parallel build jobs.
*   **NVTE_BUILD_THREADS_PER_JOB:** Control threads per job.

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch.  To verify which version is used, set:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

## Troubleshooting

### Common Issues and Solutions

1.  **ABI Compatibility Issues:**

    *   **Symptoms:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI settings.

2.  **Missing Headers or Libraries:**

    *   **Symptoms:** CMake errors about missing headers (e.g., `cudnn.h`, `cublas_v2.h`).
    *   **Solution:** Install missing development packages or set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`).

3.  **Build Resource Issues:**

    *   **Symptoms:** Compilation hangs or out-of-memory errors.
    *   **Solution:** Limit parallel builds using `MAX_JOBS=1`.

4.  **Verbose Build Logging:**

    *   For detailed logs:
        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding mask definition for PyTorch

From v1.7, all attention mask types follow the same definition where `True` means masking out the corresponding position and `False` means including that position in attention calculation.

## FP8 Convergence

FP8 has shown no significant difference in loss curves compared to BF16 training. It has been validated for accuracy on LLM tasks.

**Examples of Models Tested for Convergence:**

| Model       | Framework         | Source                                                                                                                                |
| :---------- | :---------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| T5-770M     | JAX/T5x          | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance                                    |
| MPT-1.3B    | Mosaic Composer   | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                                                             |
| GPT-5B      | JAX/Paxml        | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results                                                 |
| GPT-5B      | NeMo Framework   | Available on request                                                                                                                  |
| LLama2-7B   | Alibaba Pai       | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                                                     |
| T5-11B      | JAX/T5x          | Available on request                                                                                                                  |
| MPT-13B     | Mosaic Composer   | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8                                        |
| GPT-22B     | NeMo Framework   | Available on request                                                                                                                  |
| LLama2-70B  | Alibaba Pai       | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                                                     |
| GPT-175B    | JAX/Paxml        | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results                                                 |

## Integrations

Transformer Engine is integrated with the following frameworks:

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

See the `<CONTRIBUTING.rst>` guide for contributing guidelines.

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