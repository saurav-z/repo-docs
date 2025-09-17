<div align="center">
    <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention" width="80%">
</div>

---

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

# Flash Sparse Attention (FSA): Supercharge Your LLMs with Optimized Sparse Attention

**Flash Sparse Attention (FSA)** is a cutting-edge implementation that drastically improves the efficiency of Native Sparse Attention (NSA) for large language models (LLMs) on modern GPUs.  [Check out the original repository here](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention).

**Key Features:**

*   **Optimized NSA Implementation:** FSA provides a highly efficient, Triton-based implementation of the NSA selected attention module.
*   **Improved Performance:** Significant speedups are achieved by minimizing memory access and computation, particularly for GQA group sizes smaller than 8.
*   **Broad Hardware Compatibility:** Designed for NVIDIA Ampere and Hopper GPUs (e.g., A100, H100, H200) and supports fp16 and bf16 data types.
*   **Flexible Configuration:** Supports a wide range of GQA group sizes (1 to 16) and various head dimensions.
*   **Easy Integration:**  Simple to integrate into existing LLM training and inference pipelines.

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

*   **[2025-09, Upcoming]:** 🚀 Online profiling module, seamlessly transitioning between NSA and FSA, will be released soon.
*   **[2025-08]:** 💥 Our [Arxiv paper](https://www.arxiv.org/abs/2508.18224) is released.
*   **[2025-08]:** 🎈 Beta version of one-step decoding is released, check the code residing in [`fsa_preview`](fsa_preview).
*   **[2025-08]:** 🎉 Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Method

FSA optimizes the Native Sparse Attention (NSA) algorithm by reordering kernel loops. Instead of looping over query tokens in the outer loop and KV blocks in the inner loop (as in the original NSA implementation), FSA reverses this order. This change, along with the use of three key kernels (main, reduction, and online softmax), reduces unnecessary memory access and computation related to padded data, while avoiding `atomic` additions. This leads to significant performance improvements.

The following image visually contrasts the computation process between the original NSA kernel and the FSA main kernel:

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

FSA's speed advantage comes from dramatically reducing kernel-level memory access and calculations.

The following graph demonstrates the execution latency comparison between NSA and FSA under different GQA group sizes. The latency of FSA is normalized to 1.

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Features

FSA is optimized for the NSA selected attention module. It offers an efficient Triton-based implementation for GQA group sizes less than 8, which is common in modern LLMs. For GQA group sizes greater than or equal to 8, FSA typically reverts to the original NSA implementation to maintain performance.

FSA is tested with:

*   NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)
*   fp16 and bf16 data types
*   Head dimensions (<= 256) across query, key, and value.
*   GQA group sizes from 1 to 16.
*   Training and inference (prefill).

## Installation

**Requirements:**

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

Install FSA dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

The `FlashSparseAttention` class (in `fsa/module/FSA.py`) provides a convenient way to use FSA.  Here's an example:

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
    ).cuda()
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

This code calls the optimized kernels implemented in the `FSATopkSparseAttention` class (in `fsa/ops/FSA_topk_sparse_attention.py`).

### Train with FSA

Training with FSA is straightforward.  You primarily need to instantiate the FSA module and compute `cu_seqlens`. An example of integrating FSA into an LLM can be found in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Detailed benchmarking instructions are in [`scripts/run_unit_test.sh`]. This script provides correctness checks (forward and backward pass), performance comparisons, and memory usage analysis.

### Benchmark FSA Selected Attention Module

The optimized NSA selected attention module can be benchmarked using the commands in [`scripts/run_unit_test_sel_attn.sh`].

> **Tip:** Experiment with different `gqa`, `seqlen`, `block_size`, and `topk` parameters in the scripts for a thorough evaluation on your hardware. Benchmarking the FSA selected attention module usually results in a greater speedup.

## Performance

### Kernel Performance

Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $BK$ and top-k value $Topk$, respectively. For FSA and NSA, the execution latency is composed of compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.

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