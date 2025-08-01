# Liger Kernel: Accelerate LLM Training with Optimized Triton Kernels

**Supercharge your Large Language Model (LLM) training with Liger Kernel, achieving up to 20% faster throughput and 60% memory reduction!**  Get started today and see the impact. [![PyPI version](https://img.shields.io/pypi/v/liger-kernel?color=green)](https://pypi.org/project/liger-kernel/) [![Downloads (Stable)](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel) [![Join Our Discord](https://dcbadge.limes.pink/api/server/gpumode?style=flat)](https://discord.gg/gpumode)

[View the original repository on GitHub](https://github.com/linkedin/Liger-Kernel)

<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png" alt="Liger Kernel Banner">

Liger Kernel provides a collection of highly optimized Triton kernels specifically designed to accelerate LLM training. These kernels enable significant performance and memory efficiency improvements, allowing you to train larger models, use longer context lengths, and experiment with larger batch sizes.

**Key Features:**

*   🚀 **Performance Boost:** Increase multi-GPU training throughput by up to 20%.
*   💾 **Memory Savings:** Reduce memory usage by up to 60%.
*   🧩 **Hugging Face Compatible:** Seamlessly integrate with Hugging Face models.
*   ⚙️ **Kernel Fusion:** Leverages kernel fusion, in-place replacement, and chunking techniques for optimal performance.
*   💯 **Exact Computation:**  Ensures accurate results with no approximations.
*   📦 **Lightweight:** Minimal dependencies (Torch and Triton) to avoid dependency hell.
*   💻 **Multi-GPU Support:** Compatible with PyTorch FSDP, DeepSpeed, DDP, and more.
*   🔄 **Trainer Framework Integration:** Compatible with Axolotl, LLaMa-Factory, SFTTrainer, Hugging Face Trainer, SWIFT, oumi.
*   🌱 **Optimized Post-Training Kernels**: Up to 80% memory savings for alignment and distillation tasks (DPO, ORPO, CPO, etc.).

### Latest News
<details>
  <summary>Recent Updates</summary>

  - [2025/03/06] Blog Post:  [Peak Performance, Minimized Memory: Optimizing torchtune’s performance with torch.compile & Liger Kernel](https://pytorch.org/blog/peak-performance-minimized-memory/)
  - [2024/12/11] Release [v0.5.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.5.0): 80% more memory efficient post training losses (DPO, ORPO, CPO, etc)!
  - [2024/12/5] Release LinkedIn Engineering Blog - [Liger-Kernel: Empowering an open source ecosystem of Triton Kernels for Efficient LLM Training](https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training)
  - [2024/11/6] Release [v0.4.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.0): Full AMD support, Tech Report, Modal CI, Llama-3.2-Vision!
  - [2024/10/21] Tech Report on Arxiv: https://arxiv.org/pdf/2410.10989
  - [2024/9/6] Release v0.2.1 ([X post](https://x.com/liger_kernel/status/1832168197002510649)). 2500+ Stars, 10+ New Contributors, 50+ PRs, 50k Downloads in two weeks!
  - [2024/8/31] CUDA MODE talk, [Liger-Kernel: Real-world Triton kernel for LLM Training](https://youtu.be/gWble4FreV4?si=dxPeIchhkJ36Mbns), [Slides](https://github.com/cuda-mode/lectures?tab=readme-ov-file#lecture-28-liger-kernel)
  - [2024/8/23] Official release: check out our [X post](https://x.com/hsu_byron/status/1827072737673982056)

</details>

## Performance Benefits

| Speed Up                  | Memory Reduction           |
|---------------------------|----------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:**
> - Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> - Hugging Face models start to OOM at a 4K context length, whereas Hugging Face + Liger Kernel scales up to 16K.

## Optimize Post Training with Liger Kernel

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

Liger Kernel also provides optimized kernels for post-training tasks such as DPO, ORPO, and SimPO, reducing memory usage by up to 80%.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Installation

### Dependencies

#### CUDA

*   `torch >= 2.1.2`
*   `triton >= 2.3.0`

#### ROCm

*   `torch >= 2.5.0` Install according to the instruction in Pytorch official webpage.
*   `triton >= 3.0.0` Install from pypi. (e.g. `pip install triton==3.0.0`)

```bash
# Need to pass the url when installing
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

### Optional Dependencies

*   `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.

> **Note:**
> Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).

**Choose your installation method:**

*   **Stable Release:** `pip install liger-kernel`
*   **Nightly Release:** `pip install liger-kernel-nightly`
*   **From Source:**

```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
# Install Default Dependencies
# Setup.py will detect whether you are using AMD or NVIDIA
pip install -e .

# Setup Development Dependencies
pip install -e ".[dev]"
```

## Getting Started

Liger Kernel offers several ways to integrate into your projects:

### 1.  AutoLigerKernelForCausalLM

The easiest way to get started, this automatically patches supported models:

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Model-Specific Patching APIs

Apply optimized kernels to specific models using patching APIs:

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# Monkey-patch the model with optimized Liger kernels
apply_liger_kernel_to_llama()

# Instantiate patched model
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. Compose Your Own Model

Utilize individual Liger kernels for maximum flexibility:

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()

# fuses linear + cross entropy layers together and performs chunk-by-chunk computation to reduce memory
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")

loss = loss_fn(model.weight, input, target)
loss.backward()
```

## Examples

Explore practical applications and integration with various training frameworks:

| **Use Case**                                    | **Description**                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [**Vision-Language Model SFT**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [**Liger ORPO Trainer**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

## High-level APIs

### AutoModel

| **AutoModel Variant** | **API** |
|-----------|---------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### Patching

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| Llama4 (Text) & (Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_llama4`   | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Gemma1      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Text)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`   | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Paligemma, Paligemma2, & Paligemma2 Mix      | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`   | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ      | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL, & QVQ       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`    | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2.5-VL       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`    | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3`    |  RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Qwen3 MoE | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Phi3 & Phi3.5       | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Granite 3.0 & 3.1   | `liger_kernel.transformers.apply_liger_kernel_to_granite`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| OLMo2   | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4   | `liger_kernel.transformers.apply_liger_kernel_to_glm4`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |

## Low-level APIs

*   **Fused Linear Kernels:** Reduce memory usage by up to 80%.
*   **Optimized Kernels:**  Utilize fusion and in-place techniques for performance improvements.

### Model Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| RMSNorm                         | `liger_kernel.transformers.LigerRMSNorm`                    |
| LayerNorm                       | `liger_kernel.transformers.LigerLayerNorm`                  |
| RoPE                            | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                          | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                           | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                    | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| Fused Linear CrossEntropy       | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| Multi Token Attention           | `liger_kernel.transformers.LigerMultiTokenAttention`        |
| Softmax                         | `liger_kernel.transformers.LigerSoftmax`                    |
| Sparsemax                       | `liger_kernel.transformers.LigerSparsemax`                  |

### Alignment Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Fused Linear CPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss          | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss         | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |
| Fused Linear KTO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`     |

### Distillation Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| KLDivergence                    | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                             | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD                  | `liger_kernel.transformers.LigerFusedLinearJSD`             |
| TVD                             | `liger_kernel.transformers.LigerTVDLoss`                    |

### Experimental Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Embedding                       | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8                | `liger_kernel.transformers.experimental.matmul` |

## Contributing, Acknowledgements, and License

*   [Contributing Guidelines](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
*   [Acknowledgements](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
*   [License Information](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)

## Sponsorship and Collaboration

Liger Kernel is supported by:

*   [Glows.ai](https://platform.glows.ai/): Sponsoring NVIDIA GPUs
*   [AMD](https://www.amd.com/en.html): Providing AMD GPUs for AMD CI.
*   [Intel](https://www.intel.com/): Providing Intel GPUs for Intel CI.
*   [Modal](https://modal.com/): Free 3000 credits for NVIDIA CI.
*   [EmbeddedLLM](https://embeddedllm.com/):  Making Liger Kernel run fast and stable on AMD.
*   [HuggingFace](https://huggingface.co/): Integrating Liger Kernel into Hugging Face Transformers and TRL.
*   [Lightning AI](https://lightning.ai/): Integrating Liger Kernel into Lightning Thunder.
*   [Axolotl](https://axolotl.ai/): Integrating Liger Kernel into Axolotl.
*   [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Integrating Liger Kernel into Llama-Factory.

## CI status

<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <th style="padding: 10px;">Build</th>
    </tr>
    <tr>
        <td style="padding: 10px;">
            <div style="display: block;">
                <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml">
                    <img src="https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml/badge.svg?event=schedule" alt="Build">
                </a>
            </div>
            <div style="display: block;">
                <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml">
                    <img src="https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml/badge.svg?event=schedule" alt="Build">
                </a>
            </div>
            <div style="display: block;">
                <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml">
                    <img src="https://github.com/linkedin/Liger-Kernel/actions/workflows/intel-ci.yml/badge.svg?event=schedule" alt="Build">
                </a>
            </div>
        </td>
    </tr>
</table>

## Contact

*   For issues, create a Github ticket in this repository.
*   For open discussion, join [our discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039).
*   For formal collaboration, send an email to Yanning Chen(yannchen@linkedin.com) and Zhipeng Wang(zhipwang@linkedin.com).

## Cite this work

```bib
@inproceedings{
hsu2025ligerkernel,
title={Liger-Kernel: Efficient Triton Kernels for {LLM} Training},
author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen and Zhipeng Wang},
booktitle={Championing Open-source DEvelopment in ML Workshop @ ICML25},
year={2025},
url={https://openreview.net/forum?id=36SjAIT42G}
}