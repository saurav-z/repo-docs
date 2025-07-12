# Liger Kernel: Supercharge LLM Training with Optimized Triton Kernels

**Boost your Large Language Model (LLM) training by up to 20% and slash memory usage by 60% with Liger Kernel!**  Dive into the world of high-performance LLM training with Liger Kernel, a library of efficient Triton kernels developed by LinkedIn to accelerate your model training.  Explore the [original repository](https://github.com/linkedin/Liger-Kernel) for more details.

[![Downloads (Stable)](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel)
[![PyPI - Version](https://img.shields.io/pypi/v/liger-kernel?color=green)](https://pypi.org/project/liger-kernel)
[![Downloads (Nightly)](https://static.pepy.tech/badge/liger-kernel-nightly)](https://pepy.tech/project/liger-kernel-nightly)
[![PyPI - Version](https://img.shields.io/pypi/v/liger-kernel-nightly?color=green)](https://pypi.org/project/liger-kernel-nightly)
[![Join Our Discord](https://dcbadge.vercel.app/api/server/gpumode?style=flat)](https://discord.gg/gpumode)

<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png" alt="Liger Kernel Banner">

[Installation](#installation) | [Getting Started](#getting-started) | [Examples](#examples) | [High-level APIs](#high-level-apis) | [Low-level APIs](#low-level-apis) | [Cite our work](#cite-this-work)

## Key Features

*   **Enhanced Performance:** Accelerate multi-GPU training throughput by up to 20%.
*   **Reduced Memory Footprint:** Minimize memory usage by up to 60% for efficient training.
*   **Ease of Use:** Easily integrate into your existing workflows with one-line patching or modular components.
*   **Accuracy:** Exact computation with rigorous unit tests to maintain precision.
*   **Lightweight Dependencies:** Built on PyTorch and Triton, minimizing external library requirements.
*   **Multi-GPU Support:** Compatible with popular distributed training frameworks (PyTorch FSDP, DeepSpeed, DDP, etc.).
*   **Post-Training Optimization:**  Optimized kernels for alignment and distillation tasks, delivering up to 80% memory savings.
*   **Extensive Integration:** Works seamlessly with leading trainer frameworks like Axolotl, LLaMA-Factory, Hugging Face Trainer, and more.

## What's New
<details>
  <summary>Recent Updates</summary>

  - [2025/03/06] Joint blog post on TorchTune × Liger - [Peak Performance, Minimized Memory: Optimizing torchtune’s performance with torch.compile & Liger Kernel](https://pytorch.org/blog/peak-performance-minimized-memory/)
  - [2024/12/11] Release of [v0.5.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.5.0): 80% more memory efficient post training losses (DPO, ORPO, CPO, etc)!
  - [2024/12/5] Release of LinkedIn Engineering Blog - [Liger-Kernel: Empowering an open source ecosystem of Triton Kernels for Efficient LLM Training](https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training)
  - [2024/11/6] Release of [v0.4.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.0): Full AMD support, Tech Report, Modal CI, Llama-3.2-Vision!
  - [2024/10/21] Release of the tech report of Liger Kernel on Arxiv: https://arxiv.org/pdf/2410.10989
  - [2024/9/6] Release of v0.2.1 ([X post](https://x.com/liger_kernel/status/1832168197002510649)). 2500+ Stars, 10+ New Contributors, 50+ PRs, 50k Downloads in two weeks!
  - [2024/8/31] CUDA MODE talk, [Liger-Kernel: Real-world Triton kernel for LLM Training](https://youtu.be/gWble4FreV4?si=dxPeIchhkJ36Mbns), [Slides](https://github.com/cuda-mode/lectures?tab=readme-ov-file#lecture-28-liger-kernel)
  - [2024/8/23] Official release: check out our [X post](https://x.com/hsu_byron/status/1827072737673982056)

</details>


## Supercharge Your Model

![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)

Liger Kernel offers significant improvements in both speed and memory efficiency, allowing you to train larger models, increase batch sizes, and extend context lengths.

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:**
> - Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> - Hugging Face models start to OOM at a 4K context length, whereas Hugging Face + Liger Kernel scales up to 16K.

## Optimize Post-Training with Liger Kernel

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

Leverage optimized post-training kernels for tasks like DPO, ORPO, and SimPO, achieving up to 80% memory reduction.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Examples

Explore practical use cases to see Liger Kernel in action:

| **Use Case**                                    | **Description**                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [**Vision-Language Model SFT**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [**Liger ORPO Trainer**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

## Installation

### Dependencies

#### CUDA

*   `torch >= 2.1.2`
*   `triton >= 2.3.0`

#### ROCm

*   `torch >= 2.5.0` (Install according to PyTorch official instructions.)
*   `triton >= 3.0.0` (Install from PyPI, e.g., `pip install triton==3.0.0`)

```bash
# Need to pass the url when installing
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

### Optional Dependencies

*   `transformers >= 4.x`: Required for patching transformer models.  The minimum version depends on the specific model.

> **Note:**
> Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).

Install the stable version:

```bash
pip install liger-kernel
```

Install the nightly version:

```bash
pip install liger-kernel-nightly
```

Install from source:

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

Choose your preferred method to apply Liger kernels:

### 1. Use AutoLigerKernelForCausalLM

This is the simplest method, automatically patching supported models with default settings.

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# This wrapper automatically applies optimized kernels if the model is supported.
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Apply Model-Specific Patching APIs

Use patching APIs to integrate optimized kernels into Hugging Face models.

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# 1a. Automatically patch the model
apply_liger_kernel_to_llama()

# 1b. Customize kernel application
apply_liger_kernel_to_llama(
  rope=True,
  swiglu=True,
  cross_entropy=True,
  fused_linear_cross_entropy=False,
  rms_norm=False
)

# 2. Instantiate the patched model
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. Compose Your Own Model

Create custom models using individual kernels.

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()

# Fuses linear + cross entropy layers for memory reduction
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")

loss = loss_fn(model.weight, input, target)
loss.backward()
```

## High-level APIs

### AutoModel

| AutoModel Variant         | API                                                 |
| ------------------------- | --------------------------------------------------- |
| AutoModelForCausalLM      | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### Patching

| Model             | API                                                     | Supported Operations                                                |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| Llama4 (Text) & (Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_llama4`   | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| LLaMA 2 & 3       | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral           | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral           | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Gemma1            | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2            | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Text)     | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`   | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Paligemma, Paligemma2, & Paligemma2 Mix       | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`   | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL, & QVQ | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`    | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2.5-VL  | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`    | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3`    |  RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Qwen3 MoE | `liger_kernel_transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Phi3 & Phi3.5       | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Granite 3.0 & 3.1   | `liger_kernel.transformers.apply_liger_kernel_to_granite`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| OLMo2   | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4   | `liger_kernel.transformers.apply_liger_kernel_to_glm4`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |

## Low-level APIs

*   **Fused Linear Kernels:** Combine linear layers with loss functions, significantly reducing memory usage (up to 80%).
*   **Fusion and In-Place Techniques:** Employed for further optimization of memory and performance.

### Model Kernels

| Kernel                       | API                                                     |
| ---------------------------- | ------------------------------------------------------- |
| RMSNorm                      | `liger_kernel.transformers.LigerRMSNorm`                    |
| LayerNorm                    | `liger_kernel.transformers.LigerLayerNorm`                  |
| RoPE                         | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                       | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                        | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                 | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| Fused Linear CrossEntropy    | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| Multi Token Attention        | `liger_kernel.transformers.LigerMultiTokenAttention`        |
| Softmax                      | `liger_kernel.transformers.LigerSoftmax`                    |
| Sparsemax                    | `liger_kernel.transformers.LigerSparsemax`                  |

### Alignment Kernels

| Kernel                       | API                                                     |
| ---------------------------- | ------------------------------------------------------- |
| Fused Linear CPO Loss        | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss        | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss      | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |
| Fused Linear KTO Loss        | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`     |

### Distillation Kernels

| Kernel                       | API                                                     |
| ---------------------------- | ------------------------------------------------------- |
| KL Divergence                | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                          | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD             | `liger_kernel.transformers.LigerFusedLinearJSD`             |
| TVD                          | `liger_kernel.transformers.LigerTVDLoss`                    |

### Experimental Kernels

| Kernel                       | API                                                     |
| ---------------------------- | ------------------------------------------------------- |
| Embedding                    | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8          | `liger_kernel.transformers.experimental.matmul`       |

## Contributing, Acknowledgements, and License

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

*   For issues, create a Github ticket in this repository.
*   For open discussion, join [our discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039).
*   For formal collaboration, send an email to yannchen@linkedin.com and hning@linkedin.com

## Cite this work

```bibtex
@article{hsu2024ligerkernelefficienttriton,
      title={Liger Kernel: Efficient Triton Kernels for LLM Training},
      author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen},
      year={2024},
      eprint={2410.10989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10989},
      journal={arXiv preprint arXiv:2410.10989},
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=linkedin/Liger-Kernel&type=Date)](https://www.star-history.com/#linkedin/Liger-Kernel&Date)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>