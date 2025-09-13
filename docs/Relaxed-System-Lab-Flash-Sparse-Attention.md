<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash-Sparse-Attention Logo" width="600">
</div>

---

<div align="center">
  [![arXiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)
</div>

## Flash Sparse Attention: Revolutionizing LLM Efficiency

This repository provides the official implementation of **Flash Sparse Attention (FSA)**, a novel kernel design that significantly boosts the performance of Native Sparse Attention (NSA) for large language models (LLMs) on modern GPUs.

**Key Features:**

*   **Optimized NSA Implementation:** FSA offers a highly efficient, Triton-based implementation of NSA for faster LLM training and inference.
*   **Improved Performance:** Experience substantial speedups through lowered kernel-level memory access and computations.
*   **Compatibility:** FSA supports various NVIDIA GPUs, data types (fp16, bf16), and GQA group sizes.
*   **Easy Integration:** Seamlessly integrate FSA into your existing LLM pipelines with the provided module and example code.
*   **Comprehensive Benchmarking:** Evaluate FSA's performance with detailed benchmarking scripts.

**[Find the original repository here](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)**

## Table of Contents

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

*   **[Upcoming]** 🚀 Online profiling module will be released soon, seamlessly transitioning between NSA and FSA.
*   **[2025-08]** 💥 [Arxiv paper](https://www.arxiv.org/abs/2508.18224) released.
*   **[2025-08]** 🎈 Beta version of one-step decoding released, check the code in [`fsa_preview`](fsa_preview).
*   **[2025-08]** 🎉 Open sourced `Flash-Sparse-Attention`, providing an optimized implementation for NSA.

## Method

FSA optimizes the NSA selected attention module by exchanging the kernel loop order. FSA loops over KV blocks in the outer loop and query tokens in the inner loop, reducing unnecessary memory access and computations for padded data. This is achieved through three major kernels: (i) main kernel batches query tokens attending to the same KV block, (ii) reduction kernel accumulates attention results, and (iii) online softmax kernel.

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

🚀 FSA achieves significant speedups by lowering kernel-level memory access volume and computations.

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce9b-ce290cb792fe" />

## Features

FSA provides an optimized kernel implementation for the NSA selected attention module, offering an efficient Triton-based implementation for GQA group sizes smaller than 8. It is well-tested with:

*   NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)
*   fp16 and bf16 datatypes
*   Head dimensions up to 256
*   Various GQA group sizes (1-16)
*   Training and inference (prefill)

## Installation

**Requirements:**

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

Use the `FlashSparseAttention` class to integrate FSA into your models:

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

Training with FSA involves replacing the attention module and computing `cu_seqlens`.  See the example in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Run the benchmarking scripts to compare FSA module performance: [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh).

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module with [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh).

> [!Tip]
> Experiment with `gqa`, `seqlen`, `block_size`, and `topk` in the scripts for comprehensive benchmarking. Benchmarking the selected attention module generally yields higher speedups.

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