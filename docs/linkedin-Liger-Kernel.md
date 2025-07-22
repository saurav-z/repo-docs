# Liger Kernel: Accelerate LLM Training with Efficient Triton Kernels

**Supercharge your LLM training with Liger Kernel, achieving up to 20% faster throughput and 60% memory reduction!**  For more information, check out the [original repo](https://github.com/linkedin/Liger-Kernel).

## Key Features

*   🚀 **Performance Boost:** Increase multi-GPU training throughput by up to 20%.
*   💾 **Memory Savings:** Reduce memory usage by up to 60%, enabling larger models and context lengths.
*   🧩 **Hugging Face Compatible:** Seamlessly integrate with Hugging Face models.
*   ✨ **Optimized Kernels:** Includes fused kernels for RMSNorm, RoPE, SwiGLU, CrossEntropy, and more.
*   🛠️ **Easy Integration:** Patch your model with a single line of code or compose your own.
*   🧪 **Exact Computation:** Exact computations ensure accuracy.
*   ⚙️ **Lightweight:** Minimal dependencies (Torch and Triton).
*   💻 **Multi-GPU Support:** Compatible with PyTorch FSDP, DeepSpeed, DDP, and more.
*   🔄 **Trainer Framework Integration:** Compatible with Axolotl, LLaMa-Factory, SFTTrainer, Hugging Face Trainer, SWIFT, and oumi.
*   🚀 **Optimized Post-Training Kernels:** Up to 80% memory savings for alignment and distillation tasks (DPO, ORPO, CPO, SimPO, KTO, JSD, etc.).

## Benchmarks

| Speed Up                                                        | Memory Reduction                                                    |
| --------------------------------------------------------------- | ------------------------------------------------------------------- |
| <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png" alt="Speed up"> | <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png" alt="Memory"> |

> **Note:** Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.

## Optimize Post Training

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

Liger Kernel offers optimized post-training kernels like DPO, ORPO, SimPO, and more, reducing memory usage by up to 80%. These can be easily used as Python modules:

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Getting Started

### Installation

#### CUDA

*   `torch >= 2.1.2`
*   `triton >= 2.3.0`

#### ROCm

*   `torch >= 2.5.0` (Install from PyTorch official webpage).
*   `triton >= 3.0.0` (Install from pip: `pip install triton==3.0.0`).

```bash
# Install ROCm - Need to pass the url when installing
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

#### Optional Dependencies

*   `transformers >= 4.x`: Required for transformers models patching APIs.

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
pip install -e .
pip install -e ".[dev]"
```

### Usage

Choose the method that best suits your needs:

1.  **AutoLigerKernelForCausalLM:**  The easiest approach, automatically patches supported models:

    ```python
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
    ```

2.  **Model-Specific Patching APIs:** Swap Hugging Face models with optimized Liger Kernels.

    ```python
    import transformers
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama() # Monkey patches the model
    # Apply selective kernels
    # apply_liger_kernel_to_llama(rope=True, swiglu=True, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=False)
    model = transformers.AutoModelForCausalLM("path/to/llama/model")
    ```

3.  **Compose Your Own Model:** Utilize individual kernels for custom model architectures.

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

### AutoModel

| **AutoModel Variant**        | **API**                                          |
| ---------------------------- | ------------------------------------------------ |
| AutoModelForCausalLM         | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### Patching

| **Model**                               | **API**                                                                     | **Supported Operations**                                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Llama4 (Text) & (Multimodal)          | `liger_kernel.transformers.apply_liger_kernel_to_llama4`                  | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                     |
| LLaMA 2 & 3                             | `liger_kernel.transformers.apply_liger_kernel_to_llama`                  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| LLaMA 3.2-Vision                        | `liger_kernel.transformers.apply_liger_kernel_to_mllama`                 | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Mistral                                 | `liger_kernel.transformers.apply_liger_kernel_to_mistral`                 | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Mixtral                                 | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`                 | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Gemma1                                  | `liger_kernel.transformers.apply_liger_kernel_to_gemma`                   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                        |
| Gemma2                                  | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`                  | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                        |
| Gemma3 (Text)                           | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`             | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                        |
| Gemma3 (Multimodal)                     | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`                  | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                        |
| Paligemma, Paligemma2, & Paligemma2 Mix | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`               | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                        |
| Qwen2, Qwen2.5, & QwQ                   | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`                   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Qwen2-VL, & QVQ                        | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`                 | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Qwen2.5-VL                              | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`               | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Qwen3                                   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3`                   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Qwen3 MoE                               | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe`               | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Phi3 & Phi3.5                           | `liger_kernel.transformers.apply_liger_kernel_to_phi3`                    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                       |
| Granite 3.0 & 3.1                       | `liger_kernel.transformers.apply_liger_kernel_to_granite`                 | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss                                                             |
| OLMo2                                   | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`                   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                             |
| GLM-4                                   | `liger_kernel.transformers.apply_liger_kernel_to_glm4`                    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy                                              |

## Low-level APIs

*   **Fused Linear Kernels:** Combine linear layers with losses, reducing memory usage (up to 80% - ideal for HBM-constrained workloads).
*   **Fusion and In-Place Techniques:** Optimize memory and performance.

### Model Kernels

| **Kernel**                      | **API**                                                     |
| ------------------------------- | ------------------------------------------------------------- |
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
| ------------------------------- | ------------------------------------------------------------- |
| Fused Linear CPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss          | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss         | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |
| Fused Linear KTO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`     |

### Distillation Kernels

| **Kernel**                      | **API**                                                     |
| ------------------------------- | ------------------------------------------------------------- |
| KLDivergence                    | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                             | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD                | `liger_kernel.transformers.LigerFusedLinearJSD`             |
| TVD                             | `liger_kernel.transformers.LigerTVDLoss`                    |

### Experimental Kernels

| **Kernel**                      | **API**                                                     |
| ------------------------------- | ------------------------------------------------------------- |
| Embedding                       | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8                | `liger_kernel.transformers.experimental.matmul` |

## Examples

*   [Hugging Face Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)
*   [Lightning Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)
*   [Medusa Multi-head LLM (Retraining Phase)](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)
*   [Vision-Language Model SFT](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)
*   [Liger ORPO Trainer](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)

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
```

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>