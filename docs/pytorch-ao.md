<div align="center">

# TorchAO: Supercharge Your AI Models with PyTorch-Native Optimization

</div>

**TorchAO is a PyTorch-native model optimization framework that accelerates AI model training and inference, offering significant speedups and memory savings. Explore TorchAO on [GitHub](https://github.com/pytorch/ao) to unlock the potential of your models!**

## Key Features

*   🚀 **Float8 Training & Inference:** Achieve significant speedups without accuracy compromise.
*   🧠 **Quantization-Aware Training (QAT):** Mitigate accuracy degradation associated with quantization.
*   💾 **Post-Training Quantization (PTQ):** Optimize for int4, int8, and more with optimized kernels.
*   ⚡️ **Sparsity:** Utilize techniques like 2:4 sparsity for increased efficiency.
*   🛠️ **Integrations:** Works seamlessly with Hugging Face Transformers, vLLM, and more.

## Latest News

Stay up-to-date with the latest advancements and integrations:

*   **[June 2024]** QAT is now integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning ([docs](https://docs.axolotl.ai/docs/qat.html))!
*   **[June 2024]** Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively

<details>
  <summary>More News</summary>

*   **[Apr 2024]** Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) at 2k H100 GPU scale
*   **[Apr 2024]** TorchAO is added as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) ([docs](https://docs.vllm.ai/en/latest/features/quantization/torchao.html))!
*   **[Mar 2024]** Our [2:4 Sparsity paper](https://openreview.net/pdf?id=O5feVk7p6Y) was accepted to SLLM @ ICLR 2025!
*   **[Jan 2024]** Our [integration with GemLite and SGLang](https://pytorch.org/blog/accelerating-llm-inference/) yielded 1.1-2x faster inference with int4 and float8 quantization across different batch sizes and tensor parallel sizes
*   **[Jan 2024]** We added [1-8 bit ARM CPU kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for linear and embedding ops

</details>

## 🌅 Overview

TorchAO is a PyTorch-native model optimization framework, designed to streamline the journey from training to serving AI models. Leveraging quantization and sparsity techniques, TorchAO offers an end-to-end solution for accelerating model performance, reducing memory footprint, and improving overall efficiency. It seamlessly integrates with `torch.compile()` and `FSDP2`, and is compatible with most Hugging Face PyTorch models.

## 🚀 Quick Start

Get started with TorchAO in just a few steps:

1.  **Installation:**

    ```bash
    pip install torchao
    ```

2.  **Quantization Example:** Quantize your model to int4.

    ```python
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    quantize_(model, Int4WeightOnlyConfig(group_size=32))
    ```

    Compared to a `torch.compiled` bf16 baseline, your quantized model should be significantly smaller and faster on a single A100 GPU. See the documentation and other examples in the repo for more information.
    
    For full model setup and benchmark details, check out our [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html). Alternatively, try quantizing your favorite model using our [HuggingFace space](https://huggingface.co/spaces/pytorch/torchao-my-repo)!

## 🛠 Installation

Install the latest stable version of TorchAO using pip:

```bash
pip install torchao
```

<details>
  <summary>Additional Installation Options</summary>

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

TorchAO seamlessly integrates with leading open-source libraries:

*   Hugging Face Transformers ([Built-in inference backend](https://huggingface.co/docs/transformers/main/quantization/torchao)) and [low bit optimizers](https://github.com/huggingface/transformers/pull/31865)
*   Hugging Face Diffusers with `torch.compile` and TorchAO in a standalone repo [diffusers-torchao](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/torchao.md)
*   Hugging Face PEFT as their [quantization backend](https://huggingface.co/docs/peft/en/developer_guides/quantization#torchao-pytorch-architecture-optimization)
*   Mobius HQQ backend leveraged our int4 kernels to get [195 tok/s on a 4090](https://github.com/mobiusml/hqq#faster-inference)
*   TorchTune for our NF4 [QLoRA](https://docs.pytorch.org/torchtune/main/tutorials/qlora_finetune.html), [QAT](https://docs.pytorch.org/torchtune/main/recipes/qat_distributed.html), and [float8 quantized fine-tuning](https://github.com/pytorch/torchtune/pull/2546) recipes
*   TorchTitan for [float8 pre-training](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)
*   VLLM for LLM serving: [usage](https://docs.vllm.ai/en/latest/features/quantization/torchao.html), [detailed docs](https://docs.pytorch.org/ao/main/torchao_vllm_integration.html)
*   SGLang for LLM serving: [usage](https://docs.sglang.ai/backend/server_arguments.html#server-arguments) and the major [PR](https://github.com/sgl-project/sglang/pull/1341).
*   Axolotl for [QAT](https://docs.axolotl.ai/docs/qat.html) and [PTQ](https://docs.axolotl.ai/docs/quantize.html)

## 🔎 Inference

TorchAO delivers impressive performance gains with minimal code modifications:

*   **Int4 weight-only**: [1.89x throughput with 58.1% less memory](torchao/quantization/README.md) on Llama-3-8B
*   **Float8 dynamic quantization**: [1.54x and 1.27x speedup on Flux.1-Dev* and CogVideoX-5b respectively](https://github.com/sayakpaul/diffusers-torchao) on H100 with preserved quality
*   **Int4 + 2:4 Sparsity**: [2.37x throughput with 67.7% memory reduction](torchao/sparsity/README.md) on Llama-3-8B

Quantize your models with a single line of code (Option 1) or directly from Hugging Face using our integration (Option 2):

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

Achieve significant VRAM reduction and performance boosts while preserving model quality.  See this [step-by-step quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe) for more details. We also release pre-quantized models [here](https://huggingface.co/pytorch).

## 🚅 Training

TorchAO offers various training optimizations:

### Quantization-Aware Training

Mitigate accuracy loss from post-training quantization using QAT:

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

In collaboration with [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat), we've developed a QAT recipe that demonstrates significant accuracy improvements over traditional PTQ, recovering **96% of the accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ). For more details, please refer to the [QAT README](torchao/quantization/qat/README.md) and the [original blog](https://pytorch.org/blog/quantization-aware-training/):

### Float8

Leverage float8 dtypes for faster training:

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

Integrate float8 training with [TorchTitan's pre-training flows](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md). For details, consult these blog posts about float8 training support:
* [Accelerating Large Scale Training and Convergence with PyTorch Float8 Rowwise on Crusoe 2K H200s](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
* [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
* [Efficient Pre-training of Llama 3-like model architectures using torchtitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
* [Float8 in PyTorch](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)

### Sparse Training

Experiment with 2:4 sparsity for improved efficiency, offering a 6% end-to-end speedup on ViT-L. Full blog [here](https://pytorch.org/blog/accelerating-neural-network-training/).

```python
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-Efficient Optimizers

Reduce GPU memory consumption during training:

**1. Quantized optimizers**: Reduce optimizer state memory by 2-4x by quantizing to lower precision

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```
See detailed [benchmarks here](https://github.com/pytorch/ao/tree/main/torchao/optim).

**2. CPU offloading**:

```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```

For maximum memory savings, we support [single GPU CPU offloading](https://github.com/pytorch/ao/tree/main/torchao/optim#optimizer-cpu-offload) that efficiently moves both gradients and optimizer state to CPU memory. This approach can **reduce your VRAM requirements by 60%** with minimal impact on training speed:

## 🎥 Videos

*   [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
*   [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
*   [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
*   [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
*   [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
*   [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)

## 💬 Citation

If you find the torchao library useful, please cite it in your work:

```bibtex
@software{torchao,
  title={TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
  author={torchao},
  url={https://github.com/pytorch/ao},
  license={BSD-3-Clause},
  month={oct},
  year={2024}
}
```