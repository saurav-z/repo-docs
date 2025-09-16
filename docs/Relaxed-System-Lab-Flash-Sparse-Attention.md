<!-- Improved README for Flash-Sparse-Attention -->

<div align="center">
  <a href="https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention">
    <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash-Sparse-Attention Logo" width="600">
  </a>
</div>

---

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

## Flash-Sparse-Attention: Accelerating Large Language Models with Optimized Sparse Attention

**Flash-Sparse-Attention (FSA) revolutionizes the efficiency of Native Sparse Attention (NSA) in LLMs, offering significant speedups on modern GPUs.** This repository provides the official implementation of FSA, featuring a novel kernel design for efficient NSA across various LLMs.

**[View the original repository on GitHub](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)**

### Key Features

*   **Optimized NSA Kernel:** FSA offers an efficient Triton-based implementation for the NSA selected attention module.
*   **Superior Performance:** Achieve significant speedups over traditional NSA implementations, especially for GQA group sizes smaller than 8.
*   **Broad Hardware Compatibility:** FSA is tested and optimized for NVIDIA Ampere and Hopper GPUs.
*   **Flexible Data Type Support:** Supports fp16 and bf16 data types.
*   **Comprehensive Benchmarking:** Includes scripts for thorough performance and correctness evaluations.
*   **Easy Integration:**  Provides a user-friendly `FlashSparseAttention` module for seamless integration into existing LLM architectures.
*   **Training and Inference Ready:** Supports both training and inference (prefill) workloads.

### Table of Contents

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

*   **[2025-09, upcoming]**: 🚀 Online profiling module, which seamlessly transitions between NSA and FSA, will be released soon.
*   **[2025-08]**: 💥 Our [Arxiv paper](https://www.arxiv.org/abs/2508.18224) is released.
*   **[2025-08]**: 🎈 Beta version of one-step decoding is released, check the code residing in [`fsa_preview`](fsa_preview).
*   **[2025-08]**: 🎉 Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Method

FSA improves upon the standard Native Sparse Attention (NSA) kernel design. NSA typically loops over query tokens in the outer loop and KV blocks in the inner loop.  FSA, in contrast, inverts this loop order, processing KV blocks in the outer loop and query tokens in the inner loop. This key change allows FSA to optimize memory access and computation, reducing the overhead associated with padded data required by hardware constraints. The core FSA implementation includes three main kernels: a kernel to batch query tokens, a reduction kernel to accumulate attention results, and an online softmax kernel. This decoupling strategy minimizes unnecessary memory operations and avoids the use of `atomic` additions.

The concrete computation process comparison between NSA (left) and FSA main kernel (right) can be visualized as follows:
<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

🚀 FSA delivers significant speedups by reducing kernel-level memory access and computational load.

Execution latency comparisons between NSA and FSA:
<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Features

FSA provides an optimized kernel implementation for the NSA selected attention module. It is designed to enhance the performance of LLMs using NSA, especially for GQA group sizes commonly found in modern models.

FSA is currently well tested with:
-   NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM);
-   Datatype of fp16 and bf16;
-   The same head dimension (less than or equal to 256) across query, key, and value;
-   Varied GQA group sizes, ranging from 1 to 16;
-   Training and inference (prefill).

## Installation

### Prerequisites

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

### Installation Instructions

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

The `FlashSparseAttention` class located in `fsa/module/FSA.py` is provided for easy integration into your LLM projects.

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

The optimized kernels are accessed through the `FSATopkSparseAttention` class, which accelerates the NSA selected attention module.

### Train with FSA

Training with FSA is straightforward. You need to instantiate the FSA module and compute the `cu_seqlens` for FSA. An example of integrating FSA into an LLM can be found in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Detailed benchmarking commands for the FSA module are provided in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh). These scripts allow for correctness, performance, and memory usage comparisons.

### Benchmark FSA Selected Attention Module

The optimized NSA selected attention module, which is a major performance bottleneck, can be benchmarked using the commands in [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh).

>   **Tip:** For comprehensive benchmarking, experiment with different `gqa`, `seqlen`, `block_size`, and `topk` parameters in the provided scripts. Benchmarking the FSA selected attention module often yields a higher speedup compared to benchmarking the FSA attention module.

## Performance

### Kernel Performance

> Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $BK$ and top-k value $Topk$, respectively. For FSA and NSA, the execution latency is composed of compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

> End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.

<img width="6165" height="3093" alt="e2e_githubpic" src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" />

## Citation

If you use FSA in your research, please cite our paper:

```
@article{yan2025flashsparseattentionalternative,
  title={Flash Sparse Attention: More Efficient Natively Trainable Sparse Attention},
  author={Yan, Ran and Jiang, Youhe and Yuan, Binhang},
  journal={arXiv preprint arXiv:2508.18224},
  year={2025}
}
```

## Acknowledgments

This project builds upon the work of:

*   **NSA paper:** [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   **NSA reference implementation:** [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)