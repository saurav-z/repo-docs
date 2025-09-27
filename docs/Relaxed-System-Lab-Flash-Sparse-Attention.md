<!-- PROJECT TITLE -->
<img width="6200" height="1294" alt="Flash-Sparse-Attention" src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" />

---

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

# Flash Sparse Attention (FSA): Accelerating Native Sparse Attention for LLMs

**Flash Sparse Attention (FSA) significantly boosts the performance of Native Sparse Attention (NSA) for efficient training and inference of large language models on modern GPUs.**  Explore the [original repository](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention) for the source code and further details.

## Key Features

*   **Optimized NSA Implementation:** FSA provides a fast, Triton-based implementation specifically designed for the Native Sparse Attention (NSA) selected attention module.
*   **GQA Support:** Optimized for GQA (Grouped-Query Attention) group sizes less than 8, common in modern LLMs.
*   **Broad Hardware Compatibility:**  Tested and optimized for NVIDIA Ampere and Hopper GPUs, supporting fp16 and bf16 data types.
*   **Flexible Configuration:** Supports varied GQA group sizes (1-16), head dimensions (<= 256), and sequence lengths.
*   **Easy Integration:** Offers a straightforward `FlashSparseAttention` module for seamless integration into existing LLM architectures.
*   **Comprehensive Benchmarking:** Includes scripts for thorough performance and correctness testing.

## Key Improvements and Benefits

*   **Speedup and Efficiency:** FSA accelerates NSA by optimizing kernel-level memory access and computation.
*   **Kernel Optimization:** FSA redesigns the kernel loop order to reduce unnecessary memory access and padding calculations.
*   **End-to-end Performance Gains:** Achieve significant speedups in both training and prefill (inference) latency for state-of-the-art LLMs.

## News

*   **[Upcoming]**: Online profiling module release.
*   **[2025-08]**: Paper release on Arxiv ([https://www.arxiv.org/abs/2508.18224](https://www.arxiv.org/abs/2508.18224)).
*   **[2025-08]**: Beta version of one-step decoding in `fsa_preview`.
*   **[2025-08]**: Open-sourced Flash-Sparse-Attention.

## Method

FSA optimizes NSA by exchanging the kernel loop order. This decoupling process involves three major kernels: (i) a main kernel that processes query tokens, (ii) a reduction kernel to accumulate results, and (iii) an online softmax kernel. This architecture reduces unnecessary memory access and computation.

The following image illustrates the difference between NSA and FSA main kernels:
<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

FSA's speed advantage comes from reducing kernel-level memory access and computation.

The following image illustrates a comparison of execution latency for NSA and FSA at varying GQA group sizes:
<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Installation

**Requirements:**

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

**Installation Command:**

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

The `FlashSparseAttention` module is easy to use. Here's an example:

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

This calls the optimized kernels via `FSATopkSparseAttention`.

### Train with FSA

Training with FSA involves replacing the attention module and computing `cu_seqlens`. See [`SparseLlamaAttention`](test/train.py) for an example.

## Evaluation

### Benchmark FSA Module

Use [`scripts/run_unit_test.sh`] to benchmark the FSA module. This script compares forward/backward correctness, performance, and memory usage.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module with [`scripts/run_unit_test_sel_attn.sh`].

> [!Tip]
> Experiment with `gqa`, `seqlen`, `block_size`, and `topk` in the scripts for thorough benchmarking. Benchmarking the FSA selected attention module usually yields significant speedups.

## Performance

### Kernel Performance

Performance comparison of Triton-based FSA, NSA, and Full Attention.
($64$, $16$) / ($128$, $8$) represents block size and top-k value, respectively. For FSA and NSA, execution latency is from compressed, selected, and sliding attention. For Full Attention, it's the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

End-to-end training and prefill latency comparison of FSA, NSA, and Full Attention in state-of-the-art LLMs.

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