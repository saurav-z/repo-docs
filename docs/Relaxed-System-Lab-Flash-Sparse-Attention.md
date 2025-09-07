<div align="center">
  <img width="6200" height="1294" alt="Flash Sparse Attention" src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" />
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

## Flash Sparse Attention: Accelerating Sparse Attention for LLMs

**Flash Sparse Attention (FSA)** offers a novel kernel design for highly efficient Native Sparse Attention (NSA), significantly boosting performance for large language models (LLMs) on modern GPUs.  [Read the full paper](https://arxiv.org/abs/2508.18224) and explore the code on [GitHub](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)!

**Key Features:**

*   **Optimized NSA Implementation:** FSA provides a highly efficient, Triton-based implementation of Native Sparse Attention.
*   **Significant Speedup:**  FSA's innovative design reduces memory access and computation, leading to substantial performance gains.
*   **Compatibility:** Supports NVIDIA Ampere and Hopper GPUs, fp16/bf16 data types, various GQA group sizes (1-16), and more.
*   **Easy Integration:** FSA is designed for easy integration into existing LLM training and inference pipelines.
*   **Comprehensive Benchmarking:** Includes scripts for detailed performance and correctness comparisons.

**Table of Contents**

*   [News](#news)
*   [Method](#method)
*   [Advantages](#advantages)
*   [Features](#features)
*   [Installation](#installation)
*   [Usage](#usage)
    *   [Instantiate FSA Module](#instantiate-fsa-module)
    *   [Train with FSA](#train-with-fsa)
*   [Evaluation](#evaluation)
    *   [Benchmark FSA Module](#benchmark-fsa-module)
    *   [Benchmark FSA Selected Attention Module](#benchmark-fsa-selected-attention-module)
*   [Performance](#performance)
    *   [Kernel Performance](#kernel-performance)
    *   [End-to-end Performance](#end-to-end-performance)
*   [Citation](#citation)
*   [Acknowledgments](#acknowledgments)

## News

*   **[2025-09, upcoming]:** 🚀 Online profiling module, which seamlessly transitions between NSA and FSA, will be released soon.
*   **[2025-08]:** 💥 Our [Arxiv paper](https://www.arxiv.org/abs/2508.18224) is released.
*   **[2025-08]:** 🎈 Beta version of one-step decoding is released, check the code residing in [`fsa_preview`](fsa_preview).
*   **[2025-08]:** 🎉 Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Method

FSA rethinks the kernel loop order of the original NSA kernel design, by looping over KV blocks in the outer loop and query tokens in the inner loop. FSA comprises the following three main kernels for optimized computation: (i) a kernel that processes query tokens for a KV block, storing partial results, (ii) a reduction kernel that accumulates attention results, and (iii) an online softmax kernel. This approach reduces unnecessary memory access and computations and avoids atomic additions.

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

🚀 FSA significantly boosts speed by minimizing kernel-level memory access volume and computations.

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Features

FSA offers an optimized kernel implementation for the NSA selected attention module, improving performance for GQA group sizes smaller than 8, which are more prevalent in state-of-the-art LLMs on modern high-performance NVIDIA GPUs.

FSA has been tested with:

*   NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM);
*   fp16 and bf16 datatypes;
*   head dimensions (less than or equal to 256) across query, key, and value;
*   GQA group sizes from 1 to 16;
*   training and inference (prefill).

## Installation

**Requirements:**

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

```python
import torch
from fsa.module.fsa import FlashSparseAttention, RopeConfig

FSA = (
    FlashSparseAttention(
        hidden_size=4096,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=128,
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        topk=16,
        init_blocks=1,
        local_blocks=2,
        window_size=512,
        rope_config=RopeConfig(
            max_position_embeddings=131072,
            head_dim=128,
            rope_theta=500000,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
    )
    .cuda()
    .to(torch.bfloat16)
)
# random input
seqlens = torch.LongTensor([65536, 32768]).int().cuda()

cu_seqlens = torch.cat(
    [
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(seqlens, dim=0),
    ],
    dim=0,
).to(torch.int32)
x = torch.randn(cu_seqlens[-1], 4096, device="cuda", dtype=torch.bfloat16)

y = FSA(x, cu_seqlens)
loss = (y * torch.randn_like(y)).sum(-1).mean()
loss.backward()
```

### Train with FSA

Training with FSA involves instantiating the FSA module and computing `cu_seqlens`. See [``SparseLlamaAttention``](test/train.py) for an example.

## Evaluation

### Benchmark FSA Module

Use the provided scripts to benchmark the FSA module.  [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh) provides detailed commands.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module using commands in [``scripts/run_unit_test_sel_attn.sh``](scripts/run_unit_test_sel_attn.sh).

>   [!Tip]
>   Experiment with different `gqa`, `seqlen`, `block_size`, and `topk` arguments for thorough benchmarking on your hardware. Benchmarking the FSA selected attention module can yield higher speedups.

## Performance

### Kernel Performance

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

<img width="6165" height="3093" alt="e2e_githubpic" src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" />

## Citation

```
@article{yan2025flashsparseattentionalternative,
  title={Flash Sparse Attention: More Efficient Natively Trainable Sparse Attention},
  author={Yan, Ran and Jiang, Youhe and Yuan, Binhang},
  journal={arXiv preprint arXiv:2508.18224},
  year={2025}
}
```

## Acknowledgments

*   NSA paper: [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   NSA reference implementation: [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)