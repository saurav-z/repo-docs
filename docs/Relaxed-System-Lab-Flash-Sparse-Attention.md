<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention" width="6200" height="1294">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

# Flash Sparse Attention: Accelerating Native Sparse Attention for LLMs

**Flash Sparse Attention (FSA) provides an optimized, high-performance implementation of Native Sparse Attention (NSA), enabling faster and more efficient training and inference for large language models on modern GPUs.** ([Original Repo](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention))

## Key Features

*   **Optimized NSA Implementation:** FSA significantly improves the performance of NSA, a natively trainable sparse attention technique.
*   **Triton-Based Kernels:** Leverages Triton for efficient kernel implementations, particularly for GQA group sizes less than 8.
*   **Broad GPU Support:**  Well-tested on NVIDIA Ampere and Hopper GPUs (A100, H100, H200, etc.)
*   **Data Type Flexibility:** Supports both fp16 and bf16 data types.
*   **Versatile:** Works with varied GQA group sizes (1-16) and supports both training and inference (prefill).
*   **Comprehensive Benchmarking:** Includes scripts for detailed performance and correctness comparisons.

## News

*   **[Upcoming]**: 🚀 Online profiling module, seamlessly transitions between NSA and FSA.
*   **[2025-08]**: 💥 [Arxiv paper](https://www.arxiv.org/abs/2508.18224) released.
*   **[2025-08]**: 🎈 Beta version of one-step decoding released in [`fsa_preview`](fsa_preview).
*   **[2025-08]**: 🎉 Open sourced `Flash-Sparse-Attention`.

## Method

FSA optimizes the NSA selected attention module by rearranging kernel loop order and decoupling the computation into three main kernels: a kernel for batching query tokens, a reduction kernel, and an online softmax kernel. This approach reduces unnecessary memory access and computations, especially when dealing with padding in GQA scenarios.

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

FSA boosts speed by significantly reducing kernel-level memory access and computations.

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Installation

**Requirements:**

*   PyTorch >= 2.4
*   Triton >=3.0
*   transformers >=4.45.0
*   datasets >=3.3.0
*   accelerate >= 1.9.0
*   flash-attn ==2.6.3

**Install with:**

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

Integrating FSA into your training pipeline involves instantiating the `FlashSparseAttention` module and computing `cu_seqlens`. See [`SparseLlamaAttention`](test/train.py) for an example.

## Evaluation

### Benchmark FSA Module

Use the commands in [`scripts/run_unit_test.sh`] to compare the performance, correctness, and memory usage of the FSA module.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module using the commands in [`scripts/run_unit_test_sel_attn.sh`].

> [!Tip]
> Experiment with different `gqa`, `seqlen`, `block_size`, and `topk` values in the scripts for comprehensive benchmarking on your hardware. Benchmarking the FSA selected attention module typically yields the greatest speedups.

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