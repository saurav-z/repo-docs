<!-- Improved README.md -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention Logo" width="600">
</div>

---

<div align="center">
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

## Flash Sparse Attention: Accelerating LLMs with Optimized Sparse Attention

**Flash Sparse Attention (FSA) dramatically improves the efficiency of Native Sparse Attention (NSA) for large language models (LLMs) on modern GPUs.**

[View the original repository on GitHub](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)

**Key Features:**

*   🚀 **Optimized NSA Implementation:** FSA provides a highly optimized, Triton-based implementation for NSA selected attention, specifically targeting GQA group sizes common in LLMs.
*   💥 **Significant Performance Gains:** Achieve substantial speedups compared to traditional NSA implementations, leading to faster training and inference.
*   🎈 **Support for Leading Hardware:** FSA is optimized for NVIDIA Ampere and Hopper GPUs, supporting fp16 and bf16 data types.
*   🎉 **Flexible Configuration:** Supports a range of GQA group sizes, head dimensions, and sequence lengths, offering versatility for different LLM architectures.
*   ⚙️ **Easy Integration:** FSA is designed for easy integration into existing LLM training pipelines.
*   🧪 **Comprehensive Benchmarking:** Includes scripts for benchmarking both the FSA module and the FSA selected attention module.

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

*   **$\texttt{[2025-09, upcoming]}$:** 🚀 Online profiling module, which seamlessly transitions between NSA and FSA, will be released soon.
*   **$\texttt{[2025-08]}$:** 💥 Our [Arxiv paper](https://www.arxiv.org/abs/2508.18224) is released.
*   **$\texttt{[2025-08]}$:** 🎈 Beta version of one-step decoding is released, check the code residing in [`fsa_preview`](fsa_preview).
*   **$\texttt{[2025-08]}$:** 🎉 Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Method

FSA optimizes the NSA selected attention module by changing the kernel loop order. Instead of looping over query tokens in the outer loop and KV blocks in the inner loop (as in the original NSA), FSA loops over KV blocks in the outer loop and query tokens in the inner loop. FSA decouples the computation into three major kernels: (i) the main kernel batches query tokens that attend to the same KV block and stores the partial results to a buffer, (ii) the reduction kernel accumulates attention results for each query token, and (iii) the online softmax kernel that handles online softmax statistics computation. This reduces unnecessary memory access and computations for padded data.

The concrete computation process comparison between NSA (left) and FSA main kernel (right) can be visualized as follows:
<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

🚀 The speedup of FSA originates from significantly lowered kernel-level memory access volume and computations.

Under varied GQA group sizes, NSA hyperparameters block size $B_K=64$ and topk-k value $T=16$, 64K sequence length, 4 KV heads, the execution latency comparisons between NSA and our method are as follows (execution latency of our method is normalized to 1):
<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Features

FSA provides an optimized kernel implementation for NSA selected attention module. Without modifying NSA algorithm, FSA provides an efficient Triton-based implementation for GQA group sizes smaller than 8, which is more popular on state-of-the-art large language models (LLMs), on modern high-performance NVIDIA GPUs. For GQA group sizes larger than or equal to 8, FSA usually chooses to fall back to the original NSA implementation for better performance.

FSA is currently well tested with:
- NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM);
- Datatype of fp16 and bf16;
- The same head dimension (less than or equal to 256) across query, key, and value;
- Varied GQA group sizes, ranging from 1 to 16;
- Training and inference (prefill).

## Installation

**Requirements:**

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

Use the `FlashSparseAttention` class to integrate FSA into your projects. Here's an example:

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

The optimized kernels are provided by the [``FSATopkSparseAttention``](fsa/ops/FSA_topk_sparse_attention.py) class.

### Train with FSA

Training with FSA is straightforward. Instantiate the FSA module and compute the `cu_seqlens`. An example integration is available in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Run the benchmark for the FSA module using the commands in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh). The benchmark compares the forward and backward correctness, performance, and memory usage.

### Benchmark FSA Selected Attention Module

The optimized NSA selected attention module can be benchmarked using the scripts in [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh).

> [!Tip]
> Experiment with different `gqa`, `seqlen`, `block_size`, and `topk` values in the scripts for a comprehensive evaluation on your hardware. Benchmarking the FSA selected attention module will usually provides a higher speedup.

## Performance

### Kernel Performance

> Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $BK$ and top-k value $Topk$, respectively. For FSA and NSA, the execution latency is composed of compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

> End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.

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

*   **NSA Paper:** [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   **NSA Reference Implementation:** [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)