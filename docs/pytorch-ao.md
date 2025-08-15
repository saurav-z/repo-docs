<div align="center">

# TorchAO: Accelerate Your AI Models with PyTorch Native Optimization 🚀

</div>

**TorchAO provides a PyTorch-native model optimization framework, empowering developers to achieve significant speedups and memory savings in both training and inference.** Visit the [original repo](https://github.com/pytorch/ao) for more details and to contribute.

## Key Features

*   **Float8 Training and Inference:** Achieve significant speedups without compromising accuracy. Pre-train Llama-3.1-70B **1.5x faster** with float8 training.
*   **Quantization-Aware Training (QAT):** Mitigate accuracy degradation from quantization. Recover **77% of quantized perplexity degradation** on Llama-3.2-3B with QAT.
*   **Post-Training Quantization (PTQ):** Quantize models to int4, int8, and other formats for faster inference and reduced memory footprint. Quantize Llama-3-8B to int4 for **1.89x faster** inference with **58% less memory**.
*   **Sparsity Support:** Explore techniques like 2:4 sparsity and block sparsity to further enhance performance.
*   **Comprehensive Integrations:** Seamlessly integrates with leading open-source libraries.

## 📣 Latest News

*   **[Jun 25]** Our [TorchAO paper](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) was accepted to CodeML @ ICML 2025!
*   **[May 25]** QAT is now integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning ([docs](https://docs.axolotl.ai/docs/qat.html))!
*   **[Apr 25]** Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) at 2k H100 GPU scale
*   **[Apr 25]** TorchAO is added as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) ([docs](https://docs.vllm.ai/en/latest/features/quantization/torchao.html))!
*   **[Mar 25]** Our [2:4 Sparsity paper](https://openreview.net/pdf?id=O5feVk7p6Y) was accepted to SLLM @ ICLR 2025!
*   **[Jan 25]** Our [integration with GemLite and SGLang](https://pytorch.org/blog/accelerating-llm-inference/) yielded 1.1-2x faster inference with int4 and float8 quantization across different batch sizes and tensor parallel sizes
*   **[Jan 25]** We added [1-8 bit ARM CPU kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for linear and embedding ops

<details>
  <summary>Older news</summary>

- [Nov 24] We achieved [1.43-1.51x faster pre-training](https://pytorch.org/blog/training-using-float8-fsdp2/) on Llama-3.1-70B and 405B using float8 training
- [Oct 24] TorchAO is added as a quantization backend to HF Transformers!
- [Sep 24] We officially launched TorchAO. Check out our blog [here](https://pytorch.org/blog/pytorch-native-architecture-optimization/)!
- [Jul 24] QAT [recovered up to 96% accuracy degradation](https://pytorch.org/blog/quantization-aware-training/) from quantization on Llama-3-8B
- [Jun 24] Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively
- [Jun 24] Block sparsity [achieved 1.46x training speeedup](https://pytorch.org/blog/speeding-up-vits/) on the ViT model with <2% drop in accuracy

</details>

## 🌅 Overview

TorchAO is a PyTorch-native model optimization framework leveraging quantization and sparsity to provide an end-to-end, training-to-serving workflow
for AI models. TorchAO works out-of-the-box with `torch.compile()` and `FSDP2` across most HuggingFace PyTorch models. Key features include:
* Float8 [training](torchao/float8/README.md) and [inference](https://docs.pytorch.org/ao/main/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html) for speedups without compromising accuracy
* [MX training and inference](torchao/prototype/mx_formats/README.md), provides MX tensor formats based on native PyTorch MX dtypes (prototype)
* [Quantization-Aware Training (QAT)](torchao/quantization/qat/README.md) for mitigating quantization degradation
* [Post-Training Quantization (PTQ)](torchao/quantization/README.md) for int4, int8, fp6 etc, with matching kernels targeting a variety of backends including CUDA, ARM CPU, and XNNPACK
* [Sparsity](torchao/sparsity/README.md), includes different techniques such as 2:4 sparsity and block sparsity

Check out our [docs](https://docs.pytorch.org/ao/main/) for more details!

From the team that brought you the fast series:
* 9.5x inference speedups for Image segmentation models with [sam-fast](https://pytorch.org/blog/accelerating-generative-ai)
* 10x inference speedups for Language models with [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2)
* 3x inference speedup for Diffusion models with [sd-fast](https://pytorch.org/blog/accelerating-generative-ai-3) (new: [flux-fast](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/))
* 2.7x inference speedup for FAIR’s Seamless M4T-v2 model with [seamlessv2-fast](https://pytorch.org/blog/accelerating-generative-ai-4/)


## 🚀 Quick Start

Get started by installing TorchAO and quantizing your model weights to int4 with just a few lines of code!

```bash
pip install torchao
```

```python
from torchao.quantization import Int4WeightOnlyConfig, quantize_
quantize_(model, Int4WeightOnlyConfig(group_size=32))
```
Compared to a `torch.compiled` bf16 baseline, your quantized model should be significantly smaller and faster on a single A100 GPU:
```
int4 model size: 1.25 MB
bfloat16 model size: 4.00 MB
compression ratio: 3.2

bf16 mean time: 30.393 ms
int4 mean time: 4.410 ms
speedup: 6.9x
```
For the full model setup and benchmark details, check out our [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html). Alternatively, try quantizing your favorite model using our [HuggingFace space](https://huggingface.co/spaces/pytorch/torchao-my-repo)!

## 🛠 Installation

Install the latest stable version of TorchAO using pip:

```bash
pip install torchao
```

<details>
  <summary>Other installation options</summary>

  ```
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

TorchAO seamlessly integrates with popular AI libraries, including:

*   Hugging Face Transformers
*   Hugging Face Diffusers
*   Hugging Face PEFT
*   Mobius HQQ
*   TorchTune
*   TorchTitan
*   VLLM
*   SGLang
*   Axolotl

## 🔎 Inference

TorchAO provides easy-to-use APIs for achieving significant performance gains:

-   **Int4 weight-only:** Up to **1.89x** throughput with **58.1%** less memory on Llama-3-8B.
-   **Float8 dynamic quantization:** Achieves **1.54x** and **1.27x** speedups on Flux.1-Dev\* and CogVideoX-5b, respectively, on H100, while preserving quality.
-   **Int4 + 2:4 Sparsity:** Delivers **2.37x** throughput with **67.7%** memory reduction on Llama-3-8B.

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

For detailed guidance, consult the [step-by-step quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe). Explore our pre-quantized models [here](https://huggingface.co/pytorch).

## 🚅 Training

### Quantization-Aware Training

Improve accuracy with Quantization-Aware Training (QAT), especially for low bit-width dtypes like int4. Recover **96%** of accuracy degradation on hellaswag and **68%** of the perplexity degradation on wikitext for Llama3 compared to post-training quantization (PTQ).

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

Users can also combine LoRA + QAT to speed up training by [1.89x](https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700) compared to vanilla QAT using this [fine-tuning recipe](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py).

### Float8

Implement training recipes using scaled float8 dtypes (https://arxiv.org/abs/2209.05433). Achieve throughput speedups of up to **1.5x** on up to 512 GPU / 405B parameter count scale ([details](https://pytorch.org/blog/training-using-float8-fsdp2/)):

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

### Sparse Training

Add support for semi-structured 2:4 sparsity with **6% end-to-end speedups on ViT-L**. Full blog [here](https://pytorch.org/blog/accelerating-neural-network-training/).

```python
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-efficient optimizers

TorchAO offers two strategies to reduce optimizer memory overhead:

**1. Quantized optimizers:** Reduce optimizer state memory by 2-4x by quantizing to lower precision

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```
**2. CPU offloading**: Move optimizer state and gradients to CPU memory for maximum memory savings. This can reduce VRAM requirements by 60% with minimal impact on training speed:

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
```