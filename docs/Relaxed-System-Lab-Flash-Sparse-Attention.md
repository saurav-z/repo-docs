<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention Logo" width="600"/>
</div>

---

<div align="center">
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

## Flash Sparse Attention: Supercharge Your LLMs with Optimized Sparse Attention

This repository provides the official implementation of **Flash Sparse Attention (FSA)**, a novel and highly efficient approach to Native Sparse Attention (NSA) that significantly boosts performance for a wide array of Large Language Models (LLMs) on modern GPUs. Explore the code and details on the paper on [arXiv](https://arxiv.org/abs/2508.18224). ([Back to Original Repo](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention))

**Key Features:**

*   🚀 **Optimized NSA Implementation:** FSA leverages a novel kernel design to provide an optimized implementation for Native Sparse Attention.
*   ⚡️ **Significant Speedups:** FSA offers notable performance improvements compared to traditional NSA and other attention mechanisms.
*   🧠 **Supports Various LLMs:** Designed for use with a wide range of popular LLMs, enhancing their efficiency.
*   💻 **Triton-Based Implementation:**  Uses Triton to efficiently implement GQA group sizes smaller than 8
*   🧪 **Well-Tested:** Tested on NVIDIA Ampere and Hopper GPUs, supporting fp16/bf16 datatypes, various GQA group sizes, and both training and inference.
*   🆕 **Upcoming Features:** Includes an online profiling module for seamless transitions between NSA and FSA, and a beta version of one-step decoding.

**Table of Contents:**

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

FSA optimizes the Native Sparse Attention (NSA) selected attention module. NSA can experience performance bottlenecks due to padding requirements on matrix dimensions for matrix multiplication, particularly with smaller GQA group sizes.

FSA addresses these limitations by exchanging the kernel loop order, looping over KV blocks in the outer loop and query tokens in the inner loop. This is achieved through three main kernels:
*   Main Kernel: Batches query tokens attending to the same KV block, saving partial results.
*   Reduction Kernel: Accumulates attention results for each query token.
*   Online Softmax Kernel: Handles online softmax statistics computation.

This approach reduces memory access and computation for padded data, preventing unnecessary `atomic` additions.

**Visual Comparison:**

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

🚀 FSA achieves significant speedups by minimizing kernel-level memory access and computation.  Experiments show reduced latency with varied GQA group sizes, block sizes, sequence lengths, and KV heads.

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Features

FSA provides an optimized kernel implementation for the NSA selected attention module.

*   **Compatibility:** Designed for use with state-of-the-art LLMs, particularly with GQA group sizes less than 8. For larger group sizes, FSA defaults to the original NSA implementation for better performance.
*   **Hardware Support:** Optimized for NVIDIA Ampere and Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM).
*   **Data Types:** Supports fp16 and bf16.
*   **Head Dimensions:** Compatible with head dimensions up to 256.
*   **GQA Group Sizes:** Supports varied GQA group sizes, ranging from 1 to 16.
*   **Use Cases:** Supports both training and inference (prefill).

## Installation

**Requirements:**

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

**Installation:**

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

Here's how to use the [`FlashSparseAttention`](fsa/module/FSA.py) module:

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
        ),
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

This example demonstrates the use of the `FlashSparseAttention` module with random input data and backpropagation. Internally, the `FSATopkSparseAttention` kernel provides the optimized performance.

### Train with FSA

To integrate FSA into your LLM training workflow:

*   Instantiate the FSA module.
*   Compute the `cu_seqlens` parameter for FSA.
*   Replace the existing attention module with `FlashSparseAttention`.

See [`SparseLlamaAttention`](test/train.py) for a complete example.

## Evaluation

### Benchmark FSA Module

Benchmark the FSA module with the commands provided in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh). This script offers comparisons of forward/backward output correctness, performance, and memory usage.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module via commands in [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh).

> **Tip:** Explore varied `gqa`, `seqlen`, `block_size`, and `topk` arguments in the scripts for thorough benchmarking on your hardware! Expect higher speedups compared to benchmarking the FSA attention module.

## Performance

### Kernel Performance

Performance comparisons of the Triton-based FSA, NSA, and Full Attention kernels are shown below. Block size (BK) and top-k values (Topk) are specified in the tuple format.  The execution latency of FSA and NSA include compressed, selected, and sliding attention, while Full Attention performance reflects the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

The following images illustrate end-to-end training and prefill latency comparisons across state-of-the-art LLMs with FSA, NSA, and Full Attention:

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