# Liger Kernel: Accelerate LLM Training with Optimized Triton Kernels

<p align="center">
    <a href="https://github.com/linkedin/Liger-Kernel">
        <img src="https://img.shields.io/github/stars/linkedin/Liger-Kernel?style=social" alt="GitHub Stars">
    </a>
    <a href="https://pepy.tech/project/liger-kernel">
        <img src="https://static.pepy.tech/badge/liger-kernel" alt="Downloads (Stable)">
    </a>
    <a href="https://pypi.org/project/liger-kernel">
        <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/liger-kernel?color=green">
    </a>
    <a href="https://discord.gg/gpumode">
      <img src="https://dcbadge.limes.pink/api/server/gpumode?style=flat" alt="Join Our Discord">
    </a>
</p>

**Liger Kernel provides highly optimized Triton kernels, dramatically speeding up Large Language Model (LLM) training while reducing memory usage.**

[View the original repository on GitHub](https://github.com/linkedin/Liger-Kernel)

## Key Features

*   **Significant Performance Gains:** Boost multi-GPU training throughput by **up to 20%** and reduce memory consumption by **up to 60%**.
*   **Optimized Post-Training:** Reduce memory usage in alignment and distillation tasks by **up to 80%**.
*   **Ease of Use:** Simple one-line integration with Hugging Face models or build custom models using our modules.
*   **Comprehensive Support:** Compatible with various training frameworks and hardware setups including NVIDIA, AMD, and Intel.
*   **Exact Computation:** Ensures accuracy with rigorous unit tests and convergence testing.
*   **Lightweight:** Minimal dependencies (PyTorch and Triton).
*   **Multi-GPU Ready:** Fully compatible with PyTorch FSDP, DeepSpeed, and other multi-GPU strategies.

## What's New

*   **[2025/03/06]**:  TorchTune x Liger joint blog post on optimizing torchtune's performance.
*   **[2024/12/11]**: Release v0.5.0: Up to 80% more memory efficient post-training losses (DPO, ORPO, CPO, etc)!
*   **[2024/12/5]**: LinkedIn Engineering Blog post on Liger-Kernel.
*   **[2024/11/6]**: Release v0.4.0: Full AMD support, Tech Report, Modal CI, Llama-3.2-Vision!
*   **[2024/10/21]**: Tech report released on Arxiv: https://arxiv.org/pdf/2410.10989
*   **[2024/9/6]**: v0.2.1 released.  2500+ Stars, 10+ New Contributors, 50+ PRs, 50k Downloads in two weeks!
*   **[2024/8/31]**: CUDA MODE talk and slides.
*   **[2024/8/23]**: Official release.

## Benchmarks & Benefits

Liger Kernel empowers you to train faster with reduced memory footprint.

| Metric                | Performance Boost                      | Memory Savings                     |
| :-------------------- | :------------------------------------- | :--------------------------------- |
| End-to-End Throughput |  Up to 20% (LLaMA 3-8B)                | Up to 60% (LLaMA 3-8B)               |
| Post-Training        |  Up to 80% memory reduction               | Up to 80% memory reduction               |

> **Note:** Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.

## Optimize Post Training with Liger Kernel

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

Liger Kernel provides optimized post-training kernels like DPO, ORPO, SimPO, and more which can reduce memory usage by up to 80%. These kernels are easily used as Python modules.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Getting Started

### Installation

**Prerequisites:**

*   `torch >= 2.1.2`
*   `triton >= 2.3.0`

**CUDA Installation:**

```bash
pip install liger-kernel
```

**ROCm Installation:**

```bash
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

**Install from Source:**

```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .
pip install -e ".[dev]"
```

### Usage

1.  **AutoLigerKernelForCausalLM:**

    ```python
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained("path/to/your/model")
    ```
2.  **Model-Specific Patching APIs:**

    ```python
    import transformers
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    apply_liger_kernel_to_llama()
    model = transformers.AutoModelForCausalLM("path/to/llama/model")
    ```
3.  **Compose Your Own Model:**

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

## High-Level APIs

*   **AutoModel:** `liger_kernel.transformers.AutoLigerKernelForCausalLM`
*   **Patching:** APIs for various models including Llama 2/3, Mistral, Mixtral, Gemma, Qwen, and more. See detailed list in original README.

## Low-level APIs

*   **Model Kernels:** `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, `MultiTokenAttention`, `Softmax`, `Sparsemax` and more.
*   **Alignment Kernels:** `FusedLinearCPOLoss`, `FusedLinearDPOLoss`, `FusedLinearORPOLoss`, `FusedLinearSimPOLoss`, `FusedLinearKTOLoss` and more.
*   **Distillation Kernels:** `KLDivergence`, `JSD`, `FusedLinearJSD`, `TVD` and more.
*   **Experimental Kernels:** `Embedding`, `Matmul int2xint8`

## Examples

| Use Case                                    | Description                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [Hugging Face Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [Lightning Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [Medusa Multi-head LLM (Retraining Phase)](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [Vision-Language Model SFT](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [Liger ORPO Trainer](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

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

*   For issues, create a Github ticket in this repository
*   For open discussion, join [our discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039)
*   For formal collaboration, send an email to Yanning Chen(yannchen@linkedin.com) and Zhipeng Wang(zhipwang@linkedin.com)

## Cite this work

Biblatex entry:

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

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=linkedin/Liger-Kernel&type=Date)](https://www.star-history.com/#linkedin/Liger-Kernel&Date)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
```

Key improvements and SEO optimizations:

*   **Clear, Concise Hook:**  The opening sentence immediately conveys the value proposition.
*   **Keyword Optimization:**  Uses relevant keywords like "Triton kernels," "LLM training," "performance," and "memory."  These are incorporated naturally throughout.
*   **Informative Headings:** Uses clear and descriptive headings to structure the information.
*   **Bulleted Lists:** Key features are easy to scan.
*   **Concise Language:** Avoids unnecessary words.
*   **Call to Action:** Encourages users to engage (star, join Discord, etc.).
*   **Contextual Links:** Links are used to enhance understanding and SEO (e.g., "original repository," "benchmarks").
*   **Up-to-date information**: The "What's New" section is optimized to keep the content up to date, which is crucial for SEO.
*   **Proper formatting**: Uses markdown for readability.
*   **More detailed benchmarks**: The included benchmark section is helpful.
*   **Consistent formatting:** Used a consistent style for the table of APIs and other elements to improve readability.
*   **Star History:** Added a star history chart to increase engagement.
*   **Clear, comprehensive and well-structured:** The README is now well-organized.