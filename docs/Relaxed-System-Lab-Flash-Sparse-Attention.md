<div align="center">
    <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash-Sparse-Attention Banner" width="800">
</div>

---

<div align="center">
    [![arXiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)
</div>

# Flash Sparse Attention (FSA): Accelerating Sparse Attention for LLMs

**Flash Sparse Attention (FSA)** provides a novel kernel design, dramatically improving the efficiency of Native Sparse Attention (NSA) on modern GPUs, leading to faster training and inference for large language models. Learn more about this innovative approach on the [original repository](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention).

**Key Features:**

*   🚀 **Optimized NSA Implementation:** FSA offers a highly efficient Triton-based implementation for NSA, addressing performance bottlenecks in popular LLMs.
*   ⚡️ **Significant Speedups:** Experience substantial kernel-level memory access reduction and computational improvements.
*   ✅ **Wide Compatibility:** FSA supports NVIDIA Ampere and Hopper GPUs, fp16/bf16 data types, various GQA group sizes (1-16), and both training and inference workloads.
*   ⚙️ **Easy Integration:** The `FlashSparseAttention` module can be easily integrated into your existing LLM architectures.
*   🧪 **Comprehensive Benchmarking:** Evaluate FSA's performance with provided scripts for correctness, speed, and memory usage comparisons.

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

*   **[2025-09, upcoming]**: 🚀 Online profiling module, which seamlessly transitions between NSA and FSA, will be released soon.
*   **[2025-08]**: 💥 Our [Arxiv paper](https://www.arxiv.org/abs/2508.18224) is released.
*   **[2025-08]**: 🎈 Beta version of one-step decoding is released, check the code residing in [`fsa_preview`](fsa_preview).
*   **[2025-08]**: 🎉 Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Method

FSA optimizes the NSA selected attention module by reordering kernel loops. Instead of looping over query tokens in the outer loop and KV blocks in the inner loop (as in the original NSA), FSA reverses this order. It decouples the computation into three major kernels to reduce memory access and computations for padded data:

1.  **Main Kernel:** Batches query tokens attending to the same KV block and stores partial results.
2.  **Reduction Kernel:** Accumulates attention results for each query token.
3.  **Online Softmax Kernel:** Computes online softmax statistics.

This architecture effectively reduces unnecessary memory access and computations, while avoiding `atomic` additions for accumulating attention results.

<img src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" alt="NSA vs FSA Kernel Comparison" width="800">

## Advantages

FSA's speedup stems from significantly lowered kernel-level memory access volume and computations.

<img src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" alt="GQA Group Size Comparison" width="600">

## Features

FSA is designed to accelerate NSA selected attention modules, particularly for GQA group sizes smaller than 8, a common configuration in modern LLMs.  It is tested with:

*   NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)
*   fp16 and bf16 data types
*   Head dimensions up to 256
*   GQA group sizes from 1 to 16
*   Training and inference (prefill)

## Installation

**Prerequisites:**

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

Use the `FlashSparseAttention` class to integrate FSA into your models. Here's a simple example:

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

Under the hood, the `FSATopkSparseAttention` class is called, providing the optimized kernels that accelerate the NSA selected attention module.

### Train with FSA

Training with FSA involves instantiating the FSA module and calculating the `cu_seqlens`. Refer to [`SparseLlamaAttention`](test/train.py) for an example.

## Evaluation

### Benchmark FSA Module

Use the script [`scripts/run_unit_test.sh`] for comprehensive benchmarking of the FSA module, including correctness, performance, and memory usage comparisons.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module using [`scripts/run_unit_test_sel_attn.sh`].

> **Tip:** Experiment with different `gqa`, `seqlen`, `block_size`, and `topk` values in the scripts to evaluate performance on your hardware. Benchmarking the FSA selected attention module often yields greater speedups.

## Performance

### Kernel Performance

Performance comparison of Triton-based FSA, NSA, and Full Attention kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $BK$ and top-k value $Topk$, respectively. For FSA and NSA, the execution latency is composed of compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.

<img src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" alt="Kernel Performance Comparison" width="800">

### End-to-end Performance

End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.

<img src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" alt="End-to-End Performance Comparison" width="800">

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