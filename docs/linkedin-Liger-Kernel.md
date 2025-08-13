# Liger Kernel: Supercharge LLM Training with Blazing-Fast Triton Kernels

**Liger Kernel accelerates Large Language Model (LLM) training by up to 20% and reduces memory usage by 60% using efficient Triton kernels.**

[Link to Original Repo](https://github.com/linkedin/Liger-Kernel)

[![Downloads (Stable)](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel)
[![PyPI - Version](https://img.shields.io/pypi/v/liger-kernel?color=green)](https://pypi.org/project/liger-kernel)
[![Downloads (Nightly)](https://static.pepy.tech/badge/liger-kernel-nightly)](https://pepy.tech/project/liger-kernel-nightly)
[![PyPI - Version](https://img.shields.io/pypi/v/liger-kernel-nightly?color=green)](https://pypi.org/project/liger-kernel-nightly)
[![Join Our Discord](https://dcbadge.limes.pink/api/server/gpumode?style=flat)](https://discord.gg/gpumode)

<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png" alt="Liger Kernel Banner">

[Installation](#installation) | [Getting Started](#getting-started) | [Examples](#examples) | [High-level APIs](#high-level-apis) | [Low-level APIs](#low-level-apis) | [Cite this work](#cite-this-work)

Liger Kernel provides a collection of highly optimized Triton kernels designed to accelerate LLM training, offering significant performance and memory improvements.  It integrates seamlessly with popular frameworks, enabling researchers and developers to train larger, more complex models faster and more efficiently.

## Key Features

*   **Performance Boost:** Increase multi-GPU training throughput by up to **20%**.
*   **Memory Savings:** Reduce memory usage by up to **60%**.
*   **Optimized Post-Training Kernels:** Achieve up to **80% memory savings** for alignment and distillation tasks (DPO, ORPO, etc.).
*   **Hugging Face Compatibility:**  Drop-in replacement for key layers like RMSNorm, RoPE, SwiGLU, and CrossEntropy.
*   **Exact Computation:** No approximations, ensuring accurate results.
*   **Minimal Dependencies:**  Requires only PyTorch and Triton.
*   **Multi-GPU Support:** Works seamlessly with FSDP, DeepSpeed, and other multi-GPU setups.
*   **Framework Integration:** Compatible with Axolotl, LLaMa-Factory, SFTTrainer, Hugging Face Trainer, SWIFT, and oumi.

## Supercharge Your Model with Liger Kernel

![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)

Liger Kernel allows you to train larger models, utilize longer context lengths, and employ larger batch sizes, all while reducing memory bottlenecks.

| Speed Up | Memory Reduction |
|---|---|
| <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png" alt="Speed Up"> | <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png" alt="Memory Reduction"> |

> **Note:**  Benchmarks were conducted using LLaMA 3-8B, bf16 data type, AdamW optimizer, gradient checkpointing, and FSDP1 on 8 A100s. Liger Kernel allows models to scale to 16K context lengths.

## Optimize Post Training with Liger Kernel

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

Liger Kernel includes optimized post-training kernels for tasks such as DPO, ORPO, SimPO, and others, delivering up to 80% memory savings.  These can be easily integrated as Python modules.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Examples

| Use Case                                  | Description                                                                                    |
|-------------------------------------------|------------------------------------------------------------------------------------------------|
| [Hugging Face Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface) | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [Lightning Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)   | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [Medusa Multi-head LLM (Retraining Phase)](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa) | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [Vision-Language Model SFT](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh) | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [Liger ORPO Trainer](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py) | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

## Installation

### Dependencies

#### CUDA

*   `torch >= 2.1.2`
*   `triton >= 2.3.0`

#### ROCm

*   `torch >= 2.5.0` (Install according to the instruction in PyTorch official webpage.)
*   `triton >= 3.0.0` (Install from PyPI: `pip install triton==3.0.0`)

```bash
# Need to pass the url when installing
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

### Optional Dependencies

*   `transformers >= 4.x`:  Required for using transformers models patching APIs.  The specific model dictates the minimum version.

> **Note:** Liger Kernel kernels are compatible with the full range of hardware supported by Triton.

To install the stable version:

```bash
pip install liger-kernel
```

To install the nightly version:

```bash
pip install liger-kernel-nightly
```

To install from source:

```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .  # Install Default Dependencies (setup.py detects AMD or NVIDIA)
pip install -e ".[dev]"  # Install Development Dependencies
```

## Getting Started

There are two main ways to integrate Liger kernels:

### 1. Use AutoLigerKernelForCausalLM

This is the simplest approach, automatically patching supported models with default settings.

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# This wrapper automatically monkey-patches the model with optimized kernels.
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Apply Model-Specific Patching APIs

Patch Hugging Face models using specific APIs for more control.

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# Monkey-patch the model with optimized kernels:
apply_liger_kernel_to_llama()

# Specify exactly which kernels to apply:
apply_liger_kernel_to_llama(
  rope=True,
  swiglu=True,
  cross_entropy=True,
  fused_linear_cross_entropy=False,
  rms_norm=False
)

# Instantiate patched model:
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. Compose Your Own Model

Utilize individual [kernels](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#model-kernels) to build custom models.

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()

# Fuses linear + cross entropy layers for memory reduction.
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")

loss = loss_fn(model.weight, input, target)
loss.backward()
```

## High-level APIs

### AutoModel

| AutoModel Variant     | API                                             |
|-----------------------|-------------------------------------------------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### Patching

| Model                        | API                                                                | Supported Operations                                                |
|------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| Llama4 (Text) & (Multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_llama4`           | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| LLaMA 2 & 3                  | `liger_kernel.transformers.apply_liger_kernel_to_llama`            | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 3.2-Vision             | `liger_kernel.transformers.apply_liger_kernel_to_mllama`          | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral                      | `liger_kernel.transformers.apply_liger_kernel_to_mistral`         | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral                      | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`         | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Gemma1                       | `liger_kernel.transformers.apply_liger_kernel_to_gemma`           | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2                       | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`          | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Text)                | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`     | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Multimodal)          | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`          | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Paligemma, Paligemma2, & Paligemma2 Mix          | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`          | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`               | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL, & QVQ         | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`               | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2.5-VL         | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`               | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3` |  RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Qwen3 MoE | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Phi3 & Phi3.5                  | `liger_kernel.transformers.apply_liger_kernel_to_phi3`            | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Granite 3.0 & 3.1            | `liger_kernel.transformers.apply_liger_kernel_to_granite`         | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| OLMo2                        | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`           | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4                        | `liger_kernel.transformers.apply_liger_kernel_to_glm4`           | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |

## Low-level APIs

*   **Fused Linear Kernels:** Combine linear layers with loss functions, reducing memory usage by up to 80%.
*   Other kernels utilize fusion and in-place techniques for enhanced performance and memory optimization.

### Model Kernels

| Kernel                      | API                                                     |
|-----------------------------|---------------------------------------------------------|
| RMSNorm                     | `liger_kernel.transformers.LigerRMSNorm`                |
| LayerNorm                     | `liger_kernel.transformers.LigerLayerNorm`                |
| RoPE                        | `liger_kernel.transformers.liger_rotary_pos_emb`          |
| SwiGLU                      | `liger_kernel.transformers.LigerSwiGLUMLP`                |
| GeGLU                       | `liger_kernel.transformers.LigerGEGLUMLP`                 |
| CrossEntropy                | `liger_kernel.transformers.LigerCrossEntropyLoss`         |
| Fused Linear CrossEntropy   | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| Multi Token Attention       | `liger_kernel.transformers.LigerMultiTokenAttention`      |
| Softmax                     | `liger_kernel.transformers.LigerSoftmax`                  |
| Sparsemax                     | `liger_kernel.transformers.LigerSparsemax`                |

### Alignment Kernels

| Kernel                      | API                                                     |
|-----------------------------|---------------------------------------------------------|
| Fused Linear CPO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`     |
| Fused Linear DPO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`     |
| Fused Linear ORPO Loss      | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`    |
| Fused Linear SimPO Loss     | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`   |
| Fused Linear KTO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`   |

### Distillation Kernels

| Kernel                      | API                                                     |
|-----------------------------|---------------------------------------------------------|
| KLDivergence                | `liger_kernel.transformers.LigerKLDIVLoss`              |
| JSD                         | `liger_kernel.transformers.LigerJSD`                    |
| Fused Linear JSD            | `liger_kernel.transformers.LigerFusedLinearJSD`         |
| TVD                         | `liger_kernel.transformers.LigerTVDLoss`                |

### Experimental Kernels

| Kernel                      | API                                                     |
|-----------------------------|---------------------------------------------------------|
| Embedding                   | `liger_kernel.transformers.experimental.LigerEmbedding` |
| Matmul int2xint8            | `liger_kernel.transformers.experimental.matmul`         |

## Contributing, Acknowledgements, and License

*   [Contributing Guidelines](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
*   [Acknowledgements](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
*   [License Information](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)

## Sponsorship and Collaboration

*   [Glows.ai](https://platform.glows.ai/):  Sponsoring NVIDIA GPUs.
*   [AMD](https://www.amd.com/en.html): Providing AMD GPUs for AMD CI.
*   [Intel](https://www.intel.com/): Providing Intel GPUs for Intel CI.
*   [Modal](https://modal.com/): Free 3000 credits for our NVIDIA CI.
*   [EmbeddedLLM](https://embeddedllm.com/): Making Liger Kernel run fast and stable on AMD.
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
                <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/intel-ci.yml">
                    <img src="https://github.com/linkedin/Liger-Kernel/actions/workflows/intel-ci.yml/badge.svg?event=schedule" alt="Build">
                </a>
            </div>
        </td>
    </tr>
</table>

## Contact

*   For issues, create a Github ticket.
*   For open discussion, join [our Discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039)
*   For formal collaboration, send an email to Yanning Chen (yannchen@linkedin.com) and Zhipeng Wang (zhipwang@linkedin.com).

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
```

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
```
Key improvements and optimizations:

*   **SEO-optimized Title and Introduction:** The title is more descriptive and includes key terms ("LLM Training," "Triton Kernels"). The introduction immediately states the core benefit (speed and memory improvements).
*   **Clear Key Features:** Uses bullet points for readability and scannability.  Keywords are bolded for emphasis.
*   **Strong Call to Action:** The initial sentence acts as a strong hook, immediately telling the user the value.
*   **Concise and Actionable Installation/Usage:** Simplified explanations of installation, and getting started with auto patching.
*   **Structured Content:** Organized with clear headings and subheadings, making it easy to navigate.
*   **Links to Key Resources:** Includes links to examples, documentation, and the original repository.
*   **Concise Language:** Avoids unnecessary words and phrases.
*   **Comprehensive Table of Contents (Implicit):** The use of headings creates a table of contents structure.
*   **Emphasis on Benefits:** The "Supercharge Your Model" section and the use of images highlight the results and advantages.
*   **Updated Badges:** Added all badges present in original.
*   **Removed Redundancy:** Streamlined the text while preserving essential information.
*   **Complete API Reference:** Added tables describing the high and low-level APIs.
*   **Clear contact information and instructions on how to cite the work.**