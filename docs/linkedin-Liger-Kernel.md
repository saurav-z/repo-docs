# Liger Kernel: Supercharge LLM Training with Optimized Triton Kernels

[<img src="https://img.shields.io/github/stars/linkedin/Liger-Kernel?style=social" alt="Stars">](https://github.com/linkedin/Liger-Kernel)

**Liger Kernel accelerates Large Language Model (LLM) training by providing optimized Triton kernels, boosting throughput and reducing memory usage.**  Explore the [original repository](https://github.com/linkedin/Liger-Kernel) for more details.

## Key Features

*   **🚀 Performance Boost:** Increase multi-GPU training throughput by up to 20%.
*   **💾 Memory Efficiency:** Reduce memory usage by up to 60%.
*   **🤝 Hugging Face Compatible:** Seamlessly integrates with Hugging Face models.
*   **🛠️ Key Kernels:** Optimized kernels for RMSNorm, RoPE, SwiGLU, CrossEntropy, and more.
*   **⚡ Post-Training Optimization:** Up to 80% memory savings for alignment and distillation tasks (DPO, ORPO, CPO, etc.).
*   **🧪 Exact Computation:**  Ensuring accuracy with rigorous unit tests.
*   **📦 Lightweight:** Minimal dependencies, only Torch and Triton required.
*   **💻 Multi-GPU Support:** Compatible with PyTorch FSDP, DeepSpeed, DDP, and other multi-GPU setups.
*   **🧩 Integration with Trainer Frameworks:** Axolotl, LLaMa-Factory, SFTTrainer, Hugging Face Trainer, SWIFT, oumi and more.

## What is Liger Kernel?

Liger Kernel is a collection of highly optimized Triton kernels designed to accelerate LLM training. It provides significant performance and memory improvements by implementing efficient versions of key operations, including RMSNorm, RoPE, SwiGLU, and CrossEntropy. Liger Kernel integrates seamlessly with popular training frameworks and models, including Hugging Face, Flash Attention, PyTorch FSDP, and DeepSpeed.

## Benefits at a Glance

| Feature               | Description                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Speed Up**          | <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png" alt="Speed up"> Increase end-to-end throughput |
| **Memory Reduction**  | <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png" alt="Memory"> Reduce memory usage |

> **Note:** Benchmarks conducted on LLaMA 3-8B, with a batch size of 8, using `bf16` data type, AdamW optimizer, Gradient Checkpointing enabled, and FSDP1 on 8 A100s.  Liger Kernel enables scaling up to 16K context length compared to Hugging Face which often OOMs at 4K.

## Optimize Post-Training

Liger Kernel provides optimized post-training kernels, like DPO, ORPO, and SimPO, reducing memory usage by up to 80%.

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Getting Started

### Installation

Install the stable version:
```bash
pip install liger-kernel
```

Or, install the nightly version:
```bash
pip install liger-kernel-nightly
```

For more installation options, see the [Installation](#installation) section in the original README.

### Applying Kernels

Choose from the following methods to apply Liger Kernel:

1.  **AutoLigerKernelForCausalLM:**  The simplest method for supported models.
    ```python
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained("path/to/your/model")
    ```

2.  **Model-Specific Patching APIs:**  Provides fine-grained control.
    ```python
    import transformers
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama() # Monkey-patches the model

    model = transformers.AutoModelForCausalLM("path/to/llama/model")
    ```

3.  **Compose Your Own Model:** Utilize individual kernels for maximum flexibility.
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

## Available APIs

### AutoModel

| AutoModel Variant          | API                                                      |
| --------------------------- | -------------------------------------------------------- |
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### Patching

| Model   | API                                                     | Supported Operations                                              |
| ---------------- | ------------------------------------------------------------- | ----------------------------------------------------------------- |
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

### Low-level APIs

#### Model Kernels

| Kernel                      | API                                                     |
| --------------------------- | -------------------------------------------------------- |
| RMSNorm                     | `liger_kernel.transformers.LigerRMSNorm`                  |
| LayerNorm                   | `liger_kernel.transformers.LigerLayerNorm`                |
| RoPE                        | `liger_kernel.transformers.liger_rotary_pos_emb`          |
| SwiGLU                      | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                       | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| Fused Linear CrossEntropy   | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| Multi Token Attention       | `liger_kernel.transformers.LigerMultiTokenAttention`        |
| Softmax                     | `liger_kernel.transformers.LigerSoftmax`                    |
| Sparsemax                   | `liger_kernel.transformers.LigerSparsemax`                  |

#### Alignment Kernels

| Kernel                      | API                                                     |
| --------------------------- | -------------------------------------------------------- |
| Fused Linear CPO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss      | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss     | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |
| Fused Linear KTO Loss       | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`     |

#### Distillation Kernels

| Kernel                      | API                                                     |
| --------------------------- | -------------------------------------------------------- |
| KLDivergence                | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                         | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD            | `liger_kernel.transformers.LigerFusedLinearJSD`             |
| TVD                         | `liger_kernel.transformers.LigerTVDLoss`                    |

#### Experimental Kernels

| Kernel                      | API                                                     |
| --------------------------- | -------------------------------------------------------- |
| Embedding                   | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8            | `liger_kernel.transformers.experimental.matmul` |

## Examples

Explore practical use cases:

*   [Hugging Face Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)
*   [Lightning Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)
*   [Medusa Multi-head LLM (Retraining Phase)](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)
*   [Vision-Language Model SFT](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)
*   [Liger ORPO Trainer](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)

## Community and Support

*   Join our [Discord](https://discord.com/channels/1189498204333543425/1275130785933951039) for discussions.
*   Report issues on [GitHub](https://github.com/linkedin/Liger-Kernel/issues).

## Contributing, Acknowledgements, and License

*   [Contributing Guidelines](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
*   [Acknowledgements](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
*   [License Information](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)

## Sponsorship and Collaboration

*   [Glows.ai](https://platform.glows.ai/) - NVIDIA GPU sponsorship
*   [AMD](https://www.amd.com/en.html) - AMD GPU support for CI
*   [Intel](https://www.intel.com/) - Intel GPU support for CI
*   [Modal](https://modal.com/) - Free credits for GPU MODE IRL CI
*   [EmbeddedLLM](https://embeddedllm.com/) - AMD support
*   [HuggingFace](https://huggingface.co/) - Integration with Transformers and TRL
*   [Lightning AI](https://lightning.ai/) - Integration with Lightning Thunder
*   [Axolotl](https://axolotl.ai/) - Integration
*   [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) - Integration

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

*   For formal collaboration, send an email to Yanning Chen(yannchen@linkedin.com) and Zhipeng Wang(zhipwang@linkedin.com)

## Cite This Work

```bib
@inproceedings{
hsu2025ligerkernel,
title={Liger-Kernel: Efficient Triton Kernels for {LLM} Training},
author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen and Zhipeng Wang},
booktitle={Championing Open-source DEvelopment in ML Workshop @ ICML25},
year={2025},
url={https://openreview.net/forum?id=36SjAIT42G}
}