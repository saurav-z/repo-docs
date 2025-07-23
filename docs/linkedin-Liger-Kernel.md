# Liger Kernel: Accelerate LLM Training with Optimized Triton Kernels

[<img src="https://img.shields.io/github/stars/linkedin/Liger-Kernel?style=social" alt="Stars"/>](https://github.com/linkedin/Liger-Kernel)
[![PyPI](https://img.shields.io/pypi/v/liger-kernel?color=green)](https://pypi.org/project/liger-kernel)
[![Downloads](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel)
[<img src="https://dcbadge.vercel.app/api/server/gpumode?style=flat" alt="Discord"/>](https://discord.gg/gpumode)

**Liger Kernel empowers faster and more efficient Large Language Model (LLM) training by providing a collection of high-performance Triton kernels.**

[**View the Liger Kernel Repository**](https://github.com/linkedin/Liger-Kernel)

Liger Kernel significantly boosts multi-GPU training throughput and slashes memory usage, offering substantial performance gains for your LLM projects.

**Key Features:**

*   ✅ **Optimized Performance:** Increase multi-GPU training throughput by **up to 20%**.
*   ✅ **Reduced Memory Footprint:** Decrease memory usage by **up to 60%**, enabling larger models and context lengths.
*   ✅ **Easy Integration:** Seamlessly integrate with popular frameworks like Hugging Face Transformers and PyTorch FSDP.
*   ✅ **Exact Computation:** Maintains accuracy with exact computations, verified through rigorous testing.
*   ✅ **Wide Compatibility:** Works with multi-GPU setups (PyTorch FSDP, DeepSpeed, DDP, etc.) and supports both CUDA and ROCm.
*   ✅ **Post-Training Optimization:**  Up to 80% memory savings for alignment and distillation tasks using optimized kernels like DPO, ORPO, and SimPO.
*   ✅ **Model Support:** Comprehensive patching support for leading LLM architectures (LLaMA, Mistral, Gemma, Qwen, Phi3, etc.)

**Key Benefits:**

*   **Longer Contexts:** Train models with extended context lengths.
*   **Larger Batch Sizes:** Utilize bigger batch sizes for improved training efficiency.
*   **Massive Vocabularies:** Support models with extensive vocabulary sizes.

**Performance at a Glance:**

| Metric          | Performance Gain               |
|-----------------|--------------------------------|
| Training Speed  | Up to 20% Faster              |
| Memory Reduction| Up to 60% Less Memory Usage    |

[<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF" alt="Banner" width="100%"/>](https://github.com/linkedin/Liger-Kernel)

**Quick Start:**

1.  **Installation:**

    ```bash
    pip install liger-kernel
    ```
    or install the nightly build with
    ```bash
    pip install liger-kernel-nightly
    ```
    For CUDA
    ```bash
    pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/cu121
    ```
    For ROCm
    ```bash
    pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
    ```

2.  **Apply with a Single Line (for supported models):**

    ```python
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained("path/to/your/model")
    ```
3.  **Or, Apply Model-Specific Patching APIs:**
    ```python
    import transformers
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama()

    model = transformers.AutoModelForCausalLM("path/to/llama/model")
    ```

**Explore Examples:**

*   [Hugging Face Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)
*   [Lightning Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)
*   [Medusa Multi-head LLM (Retraining Phase)](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)
*   [Vision-Language Model SFT](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)
*   [Liger ORPO Trainer](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)

**Key APIs:**

*   **AutoModel Support:** `AutoLigerKernelForCausalLM`
*   **Patching APIs:**  `apply_liger_kernel_to_llama`, `apply_liger_kernel_to_mistral`, `apply_liger_kernel_to_qwen2`, and more (see the original repo for complete list).
*   **Low-Level Kernels:**  `LigerRMSNorm`, `LigerLayerNorm`, `liger_rotary_pos_emb`, `LigerSwiGLUMLP`, `LigerFusedLinearCrossEntropyLoss`, and many more.
*   **Alignment Kernels:** `LigerFusedLinearCPOLoss`, `LigerFusedLinearDPOLoss`, `LigerFusedLinearORPOLoss`, `LigerFusedLinearSimPOLoss`, `LigerFusedLinearKTOLoss`
*   **Distillation Kernels:** `LigerKLDIVLoss`, `LigerJSD`, `LigerFusedLinearJSD`, `LigerTVDLoss`

**Latest News:**
*   [2025/03/06] We release a joint blog post on TorchTune × Liger - [Peak Performance, Minimized Memory: Optimizing torchtune’s performance with torch.compile & Liger Kernel](https://pytorch.org/blog/peak-performance-minimized-memory/)
*   [2024/12/11] We release [v0.5.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.5.0): 80% more memory efficient post training losses (DPO, ORPO, CPO, etc)!
*   [2024/12/5] We release LinkedIn Engineering Blog - [Liger-Kernel: Empowering an open source ecosystem of Triton Kernels for Efficient LLM Training](https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training)
*   [2024/11/6] We release [v0.4.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.0): Full AMD support, Tech Report, Modal CI, Llama-3.2-Vision!
*   [2024/10/21] We have released the tech report of Liger Kernel on Arxiv: https://arxiv.org/pdf/2410.10989
*   [2024/9/6] We release v0.2.1 ([X post](https://x.com/liger_kernel/status/1832168197002510649)). 2500+ Stars, 10+ New Contributors, 50+ PRs, 50k Downloads in two weeks!
*   [2024/8/31] CUDA MODE talk, [Liger-Kernel: Real-world Triton kernel for LLM Training](https://youtu.be/gWble4FreV4?si=dxPeIchhkJ36Mbns), [Slides](https://github.com/cuda-mode/lectures?tab=readme-ov-file#lecture-28-liger-kernel)
*   [2024/8/23] Official release: check out our [X post](https://x.com/hsu_byron/status/1827072737673982056)

**Get Involved:**

*   [Contributing Guidelines](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
*   [Acknowledgements](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
*   [License Information](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)
*   [Join the Discussion](https://discord.com/channels/1189498204333543425/1275130785933951039)

**Citations:**

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

**Sponsorship and Collaboration:**

*   Glows.ai
*   AMD
*   Intel
*   Modal
*   EmbeddedLLM
*   HuggingFace
*   Lightning AI
*   Axolotl
*   Llama-Factory

---
<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>