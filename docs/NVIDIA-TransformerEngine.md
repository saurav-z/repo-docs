# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer model performance on NVIDIA GPUs with the Transformer Engine library, achieving faster training and inference with lower memory utilization.**  ([Original Repo](https://github.com/NVIDIA/TransformerEngine))

## Key Features:

*   **FP8 Support:** Accelerate training and inference with 8-bit floating-point precision on Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Utilize easy-to-use modules designed for building Transformer layers.
*   **Performance Boosts:** Benefit from optimizations like fused kernels specifically tailored for Transformer models.
*   **Broad Precision Support:** Leverage optimizations across FP16, BF16, and FP8 on NVIDIA Ampere and later architectures.
*   **Framework Agnostic:** Integrate with popular LLM frameworks and libraries through C++ and Python APIs.

## Quickstart:

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

# Create an FP8 recipe.
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

# Create an FP8 recipe.
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

For a more detailed tutorial, check out the [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

## Installation:

### System Requirements:

*   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere GPUs
*   **OS:** Linux (Official), WSL2 (Limited Support)
*   **CUDA:** 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
*   **cuDNN:** 9.3+
*   **Compiler:** GCC 9+ or Clang 10+ with C++17 support
*   **Python:** 3.12 recommended

### Installation Methods:

1.  **Docker (Recommended):**  Use pre-built images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3  # PyTorch Example
    docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.04-py3 # JAX Example
    ```

2.  **pip:**

    ```bash
    pip install --no-build-isolation transformer_engine[pytorch] # For PyTorch
    pip install --no-build-isolation transformer_engine[jax]   # For JAX
    pip install --no-build-isolation transformer_engine[pytorch,jax] # For both
    ```
    Alternatively, install from the GitHub repository:
    ```bash
    pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
    ```
    You can specify frameworks with the environment variable:
    ```bash
    NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
    ```

3.  **conda:**

    ```bash
    conda install -c conda-forge transformer-engine-torch # For PyTorch
    # JAX integration (coming soon)
    ```
4.  **Source Installation:** [See the Installation Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source)

### Troubleshooting:

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source if necessary.
2.  **Missing Headers or Libraries:** Install development packages (e.g., CUDA, cuDNN) or set environment variables like `CUDA_PATH` and `CUDNN_PATH`.
3.  **Build Resource Issues:** Limit parallel builds to resolve compilation hangs or out-of-memory errors.
    ```bash
    MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...
    ```
4.  **Verbose Build Logging:** Use verbose flags (`-v -v -v`) for detailed build logs.

## Breaking Changes:

*   **v1.7:** Padding mask definition change.  In PyTorch, `True` in the padding mask now *excludes* tokens from attention, aligning with other frameworks.

## FP8 Convergence:

Extensive testing demonstrates **no significant difference** in convergence between FP8 and BF16 training.  FP8 achieves equivalent accuracy on downstream tasks like LAMBADA and WikiText.

| Model       | Framework        | Source                                                                                                     |
|-------------|------------------|-------------------------------------------------------------------------------------------------------------|
| T5-770M     | JAX/T5x          | [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x#convergence-and-performance)       |
| MPT-1.3B    | Mosaic Composer  | [MosaicML Blog](https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1)                                                |
| GPT-5B      | JAX/Paxml        | [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results)               |
| GPT-5B      | NeMo Framework   | (Available on request)                                                                                    |
| LLama2-7B   | Alibaba Pai      | [WeChat](https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ)                                                       |
| T5-11B      | JAX/T5x          | (Available on request)                                                                                    |
| MPT-13B     | Mosaic Composer  | [Databricks Blog](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)         |
| GPT-22B     | NeMo Framework   | (Available on request)                                                                                    |
| LLama2-70B  | Alibaba Pai      | [WeChat](https://mp.weixin.qq.com/s/NQT0uKXLbXyh5031zBdeBQ)                                                       |
| GPT-175B    | JAX/Paxml        | [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax#h100-results)               |

## Integrations:

Transformer Engine is integrated with leading LLM frameworks: DeepSpeed, Hugging Face Accelerate, Lightning, MosaicML Composer, NVIDIA JAX Toolbox, NVIDIA Megatron-LM, NVIDIA NeMo Framework, Amazon SageMaker Model Parallel Library, Levanter, GPT-NeoX, Hugging Face Nanotron (coming soon), Colossal-AI (coming soon), PeriFlow (coming soon).

## Contributing:

We welcome your contributions!  Please refer to the `<CONTRIBUTING.rst>` guide for details.

## Papers:

*   [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM Tensor Parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM Sequence Parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Videos:

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72457/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)

## Previous News:

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