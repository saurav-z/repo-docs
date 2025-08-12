<div align="center">

# TorchAO: Accelerate Your AI Models with PyTorch-Native Optimization

</div>

**TorchAO empowers you to optimize your PyTorch models from training to serving, achieving significant speedups, reduced memory footprint, and improved efficiency - all with native PyTorch.**

[Visit the Original Repo](https://github.com/pytorch/ao)

**Key Features:**

*   🚀 **Float8 Training & Inference:** Achieve up to 1.5x faster training and inference speeds without sacrificing accuracy.
*   🧠 **Quantization-Aware Training (QAT):** Mitigate accuracy degradation with QAT, recovering up to 96% of accuracy lost during quantization.
*   💾 **Post-Training Quantization (PTQ):** Quantize to int4, int8, fp6, and more, with optimized kernels for CUDA, ARM CPU, and XNNPACK, reducing memory usage and increasing inference speed.
*   ✂️ **Sparsity:** Implement 2:4 sparsity and block sparsity for faster training and inference.
*   🛠️ **Integrations:** Seamlessly integrates with Hugging Face Transformers, VLLM, and other leading libraries.

## 🌟 Latest News

*   **[June 2025]** TorchAO paper accepted to CodeML @ ICML 2025!
*   **[May 2025]** QAT integration in [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning.
*   **[April 2025]** Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
*   **[April 2025]** TorchAO as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html).
*   **[March 2025]** 2:4 Sparsity paper accepted to SLLM @ ICLR 2025!
*   **[January 2025]** Integration with GemLite and SGLang led to 1.1-2x faster inference using int4/float8.
*   **[January 2025]** 1-8 bit ARM CPU kernels added for linear and embedding ops.

<details>
  <summary>Older news</summary>

  *   **[November 2024]** 1.43-1.51x faster pre-training on Llama-3.1-70B and 405B using float8 training
  *   **[October 2024]** TorchAO as a quantization backend to HF Transformers!
  *   **[September 2024]** Official TorchAO launch.
  *   **[July 2024]** QAT [recovered up to 96% accuracy degradation](https://pytorch.org/blog/quantization-aware-training/) on Llama-3-8B.
  *   **[June 2024]** Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively.
  *   **[June 2024]** Block sparsity [achieved 1.46x training speeedup](https://pytorch.org/blog/speeding-up-vits/) on the ViT model with <2% drop in accuracy
</details>

## 🌅 Overview

TorchAO is a PyTorch-native model optimization framework designed for end-to-end AI model optimization, from training to serving, utilizing quantization and sparsity techniques. It seamlessly integrates with `torch.compile()` and `FSDP2` and works out-of-the-box with most HuggingFace PyTorch models.

**Key Capabilities:**

*   **Float8 Training and Inference:** Accelerate model training and inference using float8 dtypes.
*   **MX Tensor Formats:** Provides MX tensor formats based on native PyTorch MX dtypes (prototype).
*   **Quantization-Aware Training (QAT):** Improve model accuracy through QAT.
*   **Post-Training Quantization (PTQ):** Quantize models to int4, int8, fp6, and more with optimized kernels.
*   **Sparsity:** Implement various sparsity techniques, including 2:4 and block sparsity.

Explore the [docs](https://docs.pytorch.org/ao/main/) for more details.

## 🚀 Quick Start

Get started in minutes with TorchAO:

1.  **Install TorchAO:**
    ```bash
    pip install torchao
    ```

2.  **Quantize Your Model:**

    ```python
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    quantize_(model, Int4WeightOnlyConfig(group_size=32))
    ```

    Example results (on a single A100 GPU):

    ```
    int4 model size: 1.25 MB
    bfloat16 model size: 4.00 MB
    compression ratio: 3.2

    bf16 mean time: 30.393 ms
    int4 mean time: 4.410 ms
    speedup: 6.9x
    ```

    Refer to our [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html) for full setup and benchmark details. Also, try quantizing your model using our [HuggingFace space](https://huggingface.co/spaces/pytorch/torchao-my-repo).

## 🛠 Installation

Install the latest stable version of TorchAO:

```bash
pip install torchao
```

<details>
  <summary>Other Installation Options</summary>

  ```bash
  # Nightly
  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126

  # Different CUDA versions
  pip install torchao --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
  pip install torchao --index-url https://download.pytorch.org/whl/cpu    # CPU only

  # For developers
  USE_CUDA=1 python setup.py develop
  USE_CPP=0 python setup.py develop
  ```
</details>

## 🔗 Integrations

TorchAO is integrated into several open-source libraries:

*   **Hugging Face Transformers:** Built-in inference backend and low-bit optimizers.
*   **Hugging Face Diffusers:** Best practices using `torch.compile`.
*   **Hugging Face PEFT:** Quantization backend for LoRA.
*   **Mobius HQQ:** int4 kernels resulting in 195 tok/s on a 4090.
*   **TorchTune:** NF4 QLoRA, QAT, and float8 quantized fine-tuning recipes.
*   **TorchTitan:** Float8 pre-training.
*   **VLLM:** LLM serving.
*   **SGLang:** LLM serving.
*   **Axolotl:** QAT and PTQ.

## 🔎 Inference

TorchAO provides significant performance gains with minimal code changes:

*   **Int4 weight-only:** 1.89x throughput with 58.1% less memory on Llama-3-8B.
*   **Float8 dynamic quantization:** 1.54x and 1.27x speedup on Flux.1-Dev* and CogVideoX-5b respectively on H100.
*   **Int4 + 2:4 Sparsity:** 2.37x throughput with 67.7% memory reduction on Llama-3-8B.

**Quantization Options:**

#### Option 1: Direct TorchAO API

```python
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=128, use_hqq=True))
```

#### Option 2: HuggingFace Integration

```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization.quant_api import Int4WeightOnlyConfig

# Create quantization configuration
quantization_config = TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=128, use_hqq=True))

# Load and automatically quantize
quantized_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

#### Deploy quantized models in vLLM with one command:

```shell
vllm serve pytorch/Phi-4-mini-instruct-int4wo-hqq --tokenizer microsoft/Phi-4-mini-instruct -O3
```

This quantization flow can provide a **67% VRAM reduction and a 12-20% speedup** on A100 GPUs, with model quality maintained. For more details, see this [step-by-step quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe).

## 🚅 Training

### Quantization-Aware Training

QAT can significantly improve model accuracy. We provide a QAT recipe, in collaboration with [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat), that recovers **96% of the accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ).

```python
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
from torchao.quantization.qat import QATConfig

# prepare
base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
quantize_(my_model, QATConfig(base_config, step="prepare"))

# train model (not shown)

# convert
quantize_(my_model, QATConfig(base_config, step="convert"))
```

Users can combine LoRA + QAT to speed up training by [1.89x](https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700) compared to vanilla QAT using this [fine-tuning recipe](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py).

### Float8

[torchao.float8](torchao/float8) implements training recipes with the scaled float8 dtypes, laid out in https://arxiv.org/abs/2209.05433. With ``torch.compile`` on, current results show throughput speedups of up to **1.5x on up to 512 GPU / 405B parameter count scale**:

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

Check out these blog posts for more details on our float8 training support:
* [Accelerating Large Scale Training and Convergence with PyTorch Float8 Rowwise on Crusoe 2K H200s](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
* [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
* [Efficient Pre-training of Llama 3-like model architectures using torchtitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
* [Float8 in PyTorch](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)

### Sparse Training

Add support for semi-structured 2:4 sparsity with **6% end-to-end speedups on ViT-L**:

```python
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-Efficient Optimizers

TorchAO provides solutions to reduce the memory overhead of optimizers:

**1. Quantized Optimizers:** Reduce optimizer state memory by 2-4x.

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```

**2. CPU Offloading:** Move optimizer state and gradients to CPU memory.

```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

## 🎥 Videos

*   [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
*   [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
*   [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
*   [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
*   [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
*   [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)

## 💬 Citation

```bibtex
@software{torchao,
  title={TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
  author={torchao},
  url={https://github.com/pytorch/ao},
  license={BSD-3-Clause},
  month={oct},
  year={2024}
}