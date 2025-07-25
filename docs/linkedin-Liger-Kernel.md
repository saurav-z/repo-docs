# Liger Kernel: Accelerate LLM Training with Optimized Triton Kernels

Liger Kernel is a library of high-performance Triton kernels designed to **dramatically speed up and reduce the memory footprint of Large Language Model (LLM) training**.  [Explore the original repository](https://github.com/linkedin/Liger-Kernel).

## Key Features

*   **Significant Performance Boost:**  Increase multi-GPU training throughput by **up to 20%**.
*   **Memory Efficiency:** Reduce memory usage by **up to 60%**, enabling larger models, longer context lengths, and bigger batch sizes. Post-training kernels can provide up to 80% memory savings.
*   **Ease of Use:**  Integrate Liger Kernel with your workflow, through AutoModel, patching APIs, or composing your own model.
*   **Exact Computation:**  Ensures accuracy with rigorous unit tests and convergence testing.
*   **Lightweight:**  Minimal dependencies (Torch, Triton) to avoid dependency headaches.
*   **Multi-GPU Support:**  Compatible with popular distributed training strategies (FSDP, DeepSpeed, DDP, etc.).
*   **Hugging Face Integration:** Supports patched APIs for popular models.
*   **Trainer Framework Integration**: Works with Axolotl, LLaMa-Factory, SFTTrainer, Hugging Face Trainer, SWIFT, and oumi.

## Quick Start: Supercharge Your Model

<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF" alt="Liger Kernel Banner">

With just one line of code, Liger Kernel can dramatically improve your training performance:

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:**  Benchmarks were performed using LLaMA 3-8B, bf16 data type, AdamW optimizer, Gradient Checkpointing, and FSDP1 on 8 A100s.

## Optimize Post Training with Liger Kernel

We provide optimized post training kernels like DPO, ORPO, SimPO, and more which can reduce memory usage by up to 80%. You can easily use them as python modules.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## Examples

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

*   `torch >= 2.5.0` Install according to the instruction in Pytorch official webpage.
*   `triton >= 3.0.0` Install from pypi. (e.g. `pip install triton==3.0.0`)

```bash
# Need to pass the url when installing
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

### Optional Dependencies

*   `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.

> **Note:** Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).

To install the stable version:

```bash
$ pip install liger-kernel
```

To install the nightly version:

```bash
$ pip install liger-kernel-nightly
```

To install from source:

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

Choose the best approach for your needs:

### 1.  AutoModel (Simplest)

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# Automatically patches supported models.
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Patching APIs (Customization)

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

apply_liger_kernel_to_llama()  # Applies default kernels

model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. Compose Your Own Model (Advanced)

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
|-----------------------|---------|
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

*   **Fused Linear Kernels:**  Combine linear layers with loss functions, saving up to 80% memory.
*   **Other Kernels:**  Utilize fusion and in-place operations for optimization.

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
*   For open discussion, join [our discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039)
*   For formal collaboration, send an email to Yanning Chen (yannchen@linkedin.com) and Zhipeng Wang (zhipwang@linkedin.com)

## Cite this work

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

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=linkedin/Liger-Kernel&type=Date)](https://www.star-history.com/#linkedin/Liger-Kernel&Date)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>