# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

[NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) is a powerful library designed to significantly speed up the training and inference of Transformer models, offering optimized performance with reduced memory usage.

**Key Features:**

*   **FP8 Precision Support:** Utilize 8-bit floating-point (FP8) precision on NVIDIA Hopper, Ada, and Blackwell GPUs for faster training and inference.
*   **Optimized Building Blocks:** Leverage highly optimized modules and kernels for popular Transformer architectures.
*   **Framework Agnostic API:** Integrate with your existing deep learning frameworks (PyTorch, JAX, etc.) using a user-friendly C++ API.
*   **Automatic Mixed Precision-like API:** Simplify mixed-precision training with a seamless, easy-to-use API.
*   **Optimizations Across Precisions:** Benefit from optimizations across FP16, BF16, and FP8 on supported NVIDIA GPU architectures.
*   **Broad Integrations:** Works with popular LLM frameworks like DeepSpeed, Hugging Face Accelerate, and more.

## Quickstart

Get started with Transformer Engine using these examples:

**PyTorch Example:**

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

**JAX (Flax) Example:**

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

For a more detailed tutorial, explore our  [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

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

The easiest way to begin using Transformer Engine is by utilizing Docker images. Find them on the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

Example:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
```

**2. pip Installation**

```bash
# PyTorch
pip install --no-build-isolation transformer_engine[pytorch]

# JAX
pip install --no-build-isolation transformer_engine[jax]

# Both
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

Install from GitHub:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**3. conda Installation**

```bash
# PyTorch
conda install -c conda-forge transformer-engine-torch

# JAX (coming soon)
```

**4. Source Installation**

See the detailed [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).

### Environment Variables

Configure the build process with these variables:

*   `CUDA_PATH`: CUDA installation path.
*   `CUDNN_PATH`: cuDNN installation path.
*   `CXX`: C++ compiler path.
*   `NVTE_FRAMEWORK`: Frameworks to build for (e.g., `pytorch,jax`).
*   `MAX_JOBS`: Limit parallel build jobs.
*   `NVTE_BUILD_THREADS_PER_JOB`: Threads per build job.

### Compiling with FlashAttention

Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch, with FlashAttention-3 prioritized if present.

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
```

**Note:** FlashAttention-2 compilation can be resource-intensive, potentially causing out-of-memory errors. Set `MAX_JOBS=1` to potentially resolve this.

## Troubleshooting

### Common Issues and Solutions

1.  **ABI Compatibility Issues:**

    *   **Symptom:** `ImportError` with undefined symbols.
    *   **Solution:** Ensure PyTorch and Transformer Engine use the same C++ ABI. Rebuild PyTorch if necessary.
2.  **Missing Headers or Libraries:**

    *   **Symptom:** CMake errors about missing headers (e.g., `cudnn.h`).
    *   **Solution:** Install development packages or set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`). Also check `CXX` for the compiler path.
3.  **Build Resource Issues:**

    *   **Symptom:** Compilation hangs, system freezes, or out-of-memory errors.
    *   **Solution:** Limit parallel builds using `MAX_JOBS=1` and `NVTE_BUILD_THREADS_PER_JOB=1`.
4.  **Verbose Build Logging:**

    *   **Command:**

        ```bash
        cd transformer_engine
        pip install -v -v -v --no-build-isolation .
        ```

## Breaking Changes

### v1.7: Padding Mask Definition for PyTorch

Since v1.7, the padding mask in the PyTorch implementation uses the same definition as other frameworks, where `True` masks out (excludes) and `False` includes the corresponding position.

## FP8 Convergence

FP8 has been rigorously tested, and results show **no significant difference** in loss curves compared to BF16 training. Accuracy has also been validated for downstream LLM tasks.  The following models have been tested for convergence:

| Model        | Framework        | Source                                                                                                  |
|--------------|------------------|---------------------------------------------------------------------------------------------------------|
| T5-770M      |  JAX/T5x         | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance|
| MPT-1.3B     |  Mosaic Composer | https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1                                              |
| GPT-5B       |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |
| GPT-5B       |  NeMo Framework  | Available on request                                                                                    |
| LLama2-7B    |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
| T5-11B       |  JAX/T5x         | Available on request                                                                                    |
| MPT-13B      |  Mosaic Composer | https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8         |
| GPT-22B      |  NeMo Framework  | Available on request                                                                                    |
| LLama2-70B   |  Alibaba Pai     | https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ                                                       |
| GPT-175B     |  JAX/Paxml       | https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results               |

## Integrations

Transformer Engine is integrated with these LLM frameworks:

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

Contributions are welcome! Follow the guidelines in the  `<CONTRIBUTING.rst>`  guide.

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
```
Key improvements and explanations:

*   **SEO Optimization:**  Used keywords like "Transformer Engine," "NVIDIA GPUs," "FP8," "Transformer models," "accelerate," "training," and "inference" in the title, headings, and body text.  This helps search engines understand the content.
*   **Concise Hook:** The first sentence provides a clear value proposition.
*   **Clear Headings:**  Organized the README into logical sections with descriptive headings.
*   **Bulleted Key Features:**  Uses bullet points to make the core benefits immediately understandable.
*   **Detailed Installation:** Provides comprehensive installation instructions, including Docker, pip, conda, and source installation, with troubleshooting steps.
*   **Code Examples:**  Includes concise, runnable code examples for both PyTorch and JAX (Flax) to get users started quickly.
*   **Troubleshooting Section:** Addresses common installation and usage issues.
*   **Breaking Changes Section:** Highlights important changes that users should be aware of.
*   **FP8 Convergence Information:**  Provides details on the effectiveness of FP8, including tested models.
*   **Integration List:**  Lists all integrations to make it easier for users to see if they can use the library with their preferred frameworks.
*   **Well-formatted:** Uses consistent formatting for readability.
*   **Links to Resources:** Links to the original repo and other relevant resources.
*   **Complete and Accurate:** Correctly incorporated all the existing information from the original README, including all the news and references.
*   **Removes Redundancy:** Removed unnecessary introductory text and consolidated information where appropriate.