<!-- Improved README - Flash Sparse Attention -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention Logo" width="600">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

## Flash Sparse Attention: Supercharge LLM Performance with Efficient Sparse Attention

**Flash Sparse Attention (FSA) provides a highly optimized implementation of Native Sparse Attention (NSA), significantly boosting the performance of large language models on modern GPUs.**  For the full details, see the original repository: [https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention).

**Key Features:**

*   **Optimized NSA Implementation:** FSA leverages a novel kernel design to optimize Native Sparse Attention, leading to significant speedups.
*   **Triton-Based Kernels:**  Utilizes Triton for efficient kernel implementations, especially for GQA group sizes less than 8.
*   **Broad GPU Compatibility:**  Well-tested and optimized for NVIDIA Ampere and Hopper GPUs (e.g., A100, H100, H200).
*   **Data Type Support:** Compatible with fp16 and bf16 data types.
*   **Flexible GQA Support:**  Supports various GQA group sizes, from 1 to 16.
*   **Easy Integration:**  Provides a `FlashSparseAttention` module for seamless integration into existing LLM architectures.

## Core Concepts

### Method

FSA optimizes NSA by changing the kernel loop order. Instead of looping over query tokens in the outer loop and KV blocks in the inner loop, FSA loops over KV blocks in the outer loop and query tokens in the inner loop. This reduces unnecessary memory access and computations, particularly for padding required by NSA kernels when GQA group sizes are small.

*   **Main Kernel:** Batches query tokens that attend to the same KV block and stores partial results.
*   **Reduction Kernel:** Accumulates attention results for each query token.
*   **Online Softmax Kernel:** Handles online softmax statistics computation.

### Advantages

FSA achieves speedups by minimizing kernel-level memory access and computations. The paper highlights significant performance improvements over the original NSA implementation, especially for common LLM configurations.

## News

*   **Upcoming:** Online profiling module for seamless transitions between NSA and FSA.
*   **August 2025:** Arxiv paper released.
*   **August 2025:** Beta version of one-step decoding is released, check the code residing in `fsa_preview`.
*   **August 2025:** Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Installation

To get started, make sure you have the following installed:

*   PyTorch >= 2.4
*   Triton >= 3.0
*   transformers >= 4.45.0
*   datasets >= 3.3.0
*   accelerate >= 1.9.0
*   flash-attn == 2.6.3

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

Here's a basic example of how to use the `FlashSparseAttention` module:

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

Easily integrate FSA into your LLM training pipelines by replacing the attention module.  Example is provided in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Detailed benchmarking instructions are available in `scripts/run_unit_test.sh`.  This includes correctness, performance, and memory usage comparisons.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module (the major system bottleneck) using the commands in `scripts/run_unit_test_sel_attn.sh`.

>   **Tip:** Experiment with `gqa`, `seqlen`, `block_size`, and `topk` in the scripts for comprehensive benchmarking.  The FSA selected attention module often provides the most significant speedup.

## Performance

### Kernel Performance

<img src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" alt="Kernel Performance" width="700">

### End-to-end Performance

<img src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" alt="End-to-End Performance" width="700">

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

*   NSA Paper: [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   NSA Reference Implementation: [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)