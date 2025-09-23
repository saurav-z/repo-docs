<!-- Improved README with SEO and Summary -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention" width="600">
  <br>
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

## Flash Sparse Attention: Accelerating LLMs with Optimized Sparse Attention

**Flash Sparse Attention (FSA) dramatically enhances the performance of natively trainable sparse attention (NSA) for modern LLMs.** [View the original repository](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention).

**Key Features:**

*   **Optimized NSA Implementation:** Provides a high-performance, Triton-based implementation of NSA selected attention, particularly beneficial for GQA group sizes common in state-of-the-art LLMs.
*   **Significant Speedups:** Achieves considerable kernel-level and end-to-end performance improvements compared to standard NSA implementations.
*   **Compatibility:** Tested on NVIDIA Ampere and Hopper GPUs (e.g., A100, H100), supporting fp16 and bf16 data types.
*   **Flexible GQA Support:** Supports various GQA group sizes, from 1 to 16.
*   **Easy Integration:**  Provides a user-friendly `FlashSparseAttention` module for straightforward integration into existing LLM architectures.
*   **Comprehensive Benchmarking:** Includes detailed benchmarking scripts for correctness, performance, and memory usage comparisons.

**[News](#news)** | **[Method](#method)** | **[Advantages](#advantages)** | **[Features](#features)** | **[Installation](#installation)** | **[Usage](#usage)** | **[Evaluation](#evaluation)** | **[Performance](#performance)** | **[Citation](#citation)** | **[Acknowledgments](#acknowledgments)**

## News

*   **[Upcoming]**: Online profiling module, offering a seamless transition between NSA and FSA.
*   **[August 2025]**: Published [Arxiv paper](https://www.arxiv.org/abs/2508.18224).
*   **[August 2025]**: Released beta version of one-step decoding in [`fsa_preview`](fsa_preview).
*   **[August 2025]**: Open-sourced `Flash-Sparse-Attention`, providing an optimized implementation for NSA.

## Method

FSA optimizes the NSA selected attention module by exchanging the kernel loop order of original NSA kernel design, looping over KV blocks in the outer loop and loops over query tokens in the inner loop. FSA optimizes performance through three main kernels: (i) the main kernel batches query tokens that attend to the same KV block and stores the partial results to a buffer, (ii) the reduction kernel accumulates attention results for each query token, and (iii) the online softmax kernel that handles online softmax statistics computation.

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

FSA delivers performance gains by significantly reducing kernel-level memory access and computation.

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

Install dependencies:

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
Under the hood, the [``FSATopkSparseAttention``](fsa/ops/FSA_topk_sparse_attention.py) class is called, provding the optimized kernels that accelerate the NSA selected attention module.

### Train with FSA

Training with FSA can be esaily achieved by replacing the attention module. The only thing you may need to handle is to instantiate the FSA module, and compute the ``cu_seqlens`` for FSA. We provide an example on how to insert FSA into a LLM in [``SparseLlamaAttention``](test/train.py).

## Evaluation

### Benchmark FSA Module

Detailed commands for benchmarking the FSA module are in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh). The benchmarking includes correctness, performance, and memory usage comparisons.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module with the commands in [``scripts/run_unit_test_sel_attn.sh``](scripts/run_unit_test_sel_attn.sh).

> **Tip:** Experiment with `gqa`, `seqlen`, `block_size`, and `topk` parameters in the scripts for comprehensive benchmarking!

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