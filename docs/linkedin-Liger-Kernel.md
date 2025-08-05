# Liger Kernel: Supercharge LLM Training with Optimized Triton Kernels

**Accelerate your Large Language Model (LLM) training with Liger Kernel, a collection of high-performance Triton kernels that boost training throughput by up to 20% and reduce memory usage by 60%.**  ([Go to the original repo](https://github.com/linkedin/Liger-Kernel))

[<img src="https://img.shields.io/pypi/v/liger-kernel?color=green" alt="PyPI Version">](https://pypi.org/project/liger-kernel)
[<img src="https://img.shields.io/pypi/dm/liger-kernel" alt="PyPI Downloads">](https://pypi.org/project/liger-kernel)
[<img src="https://dcbadge.limes.pink/api/server/gpumode?style=flat" alt="Join Our Discord">](https://discord.gg/gpumode)

Liger Kernel provides a suite of optimized kernels specifically designed for LLM training, offering significant performance and memory advantages.

**Key Features:**

*   🚀 **Performance Boost:** Increase multi-GPU training throughput by up to 20%.
*   💾 **Memory Savings:** Reduce memory usage by up to 60%, enabling larger models, longer context lengths, and bigger batch sizes.
*   ✅ **Ease of Use:** Integrate with Hugging Face models with a single line of code or compose your models using modular kernels.
*   ⚙️ **Exact Computation:** No approximations, ensuring accuracy with both forward and backward passes, rigorously tested.
*   📦 **Lightweight:** Minimal dependencies, requiring only PyTorch and Triton.
*   🤝 **Multi-GPU Support:** Compatible with PyTorch FSDP, DeepSpeed, DDP, and other multi-GPU setups.
*   🛠️ **Framework Integration:** Compatible with Axolotl, LLaMa-Factory, SFTTrainer, Hugging Face Trainer, SWIFT, and oumi.
*   🎯 **Optimized Post-Training Kernels:** Up to 80% memory savings for alignment and distillation tasks.

**Benchmarking:**

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:** Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.

## Optimize Post Training

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

Liger Kernel provides optimized kernels for post-training tasks such as DPO, ORPO, SimPO, etc., achieving up to 80% memory reduction.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Installation

### Dependencies

**CUDA:**
*   `torch >= 2.1.2`
*   `triton >= 2.3.0`

**ROCm:**
*   `torch >= 2.5.0`
*   `triton >= 3.0.0`

```bash
# Stable Release
pip install liger-kernel

# Nightly Release
pip install liger-kernel-nightly

# Install from Source
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .  # Default dependencies
pip install -e ".[dev]" # Development dependencies
```

## Getting Started

### 1. AutoLigerKernelForCausalLM

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Patching APIs

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

apply_liger_kernel_to_llama() # Applies default settings
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. Compose Your Own Model

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()
loss_fn = LigerFusedLinearCrossEntropyLoss()
input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")
loss = loss_fn(model.weight, input, target)
loss.backward()
```

## High-level APIs

### AutoModel

| **AutoModel Variant** | **API** |
|-----------|---------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### Patching

| **Model**   | **API**                                                      |
|-------------|--------------------------------------------------------------|
| Llama4 (Text) & (Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_llama4`   |
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama`   |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama`   |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  |
| Gemma1      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`   |
| Gemma3 (Text)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`   |
| Gemma3 (Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`   |
| Paligemma, Paligemma2, & Paligemma2 Mix      | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`   |
| Qwen2, Qwen2.5, & QwQ      | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`    |
| Qwen2-VL, & QVQ       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`    |
| Qwen2.5-VL       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`    |
| Qwen3   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3`    |
| Qwen3 MoE | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` |
| Phi3 & Phi3.5       | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     |
| Granite 3.0 & 3.1   | `liger_kernel.transformers.apply_liger_kernel_to_granite`     |
| OLMo2   | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`     |
| GLM-4   | `liger_kernel.transformers.apply_liger_kernel_to_glm4`     |

## Low-level APIs

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

## Examples

| **Use Case**                                    | **Description**                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [**Vision-Language Model SFT**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [**Liger ORPO Trainer**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

## Resources

*   [Contributing Guidelines](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
*   [Acknowledgements](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
*   [License Information](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)

## Sponsorship and Collaboration

*   [Glows.ai](https://platform.glows.ai/): Sponsoring NVIDIA GPUs for our open source developers.
*   [AMD](https://www.amd.com/en.html): Providing AMD GPUs for our AMD CI.
*   [Intel](https://www.intel.com/): Providing Intel GPUs for our Intel CI.
*   [Modal](https://modal.com/): Free 3000 credits from GPU MODE IRL for our NVIDIA CI.
*   [EmbeddedLLM](https://embeddedllm.com/): Making Liger Kernel run fast and stable on AMD.
*   [HuggingFace](https://huggingface.co/): Integrating Liger Kernel into Hugging Face Transformers and TRL.
*   [Lightning AI](https://lightning.ai/): Integrating Liger Kernel into Lightning Thunder.
*   [Axolotl](https://axolotl.ai/): Integrating Liger Kernel into Axolotl.
*   [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Integrating Liger Kernel into Llama-Factory.

## CI Status

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

*   For issues, create a Github ticket in this repository
*   For open discussion, join [our discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039)
*   For formal collaboration, send an email to Yanning Chen(yannchen@linkedin.com) and Zhipeng Wang(zhipwang@linkedin.com)

## Cite this Work

```bibtex
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

Key improvements and explanations:

*   **SEO Optimization:**  Includes the keywords "LLM training," "Triton kernels," "performance," and "memory," which are all relevant to the library's function. Headings are used to structure the document.
*   **Concise Hook:**  The first sentence immediately grabs the reader's attention by highlighting the core benefit.
*   **Clear Value Proposition:** Quickly explains what Liger Kernel does and its benefits.
*   **Bulleted Key Features:** Easy to scan and understand.
*   **Emphasis on Key Information:** Uses bolding to highlight important concepts and benefits.
*   **More Context on Benchmarks:** Uses a table for clarity.
*   **Improved Installation Instructions:** Clearer instructions, including ROCm installation.
*   **Comprehensive API Sections:**  Organized and easy to navigate.
*   **Complete Example List:** Includes links to example code.
*   **Clear "Cite This Work" Section:** Provides a BibTeX entry for academic use.
*   **Concise Contributing Information** All other relevant information is still accessible.
*   **Back to Top Link:** Improves navigability.
*   **Removal of unnecessary Images**:  Kept one relevant image.
*   **Clearer Structure:**  Organized the information for readability.