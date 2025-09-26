# Flash Sparse Attention: Accelerating Large Language Models with Optimized Sparse Attention

**Flash Sparse Attention (FSA)** offers a novel kernel design for efficiently implementing Native Sparse Attention (NSA), significantly boosting the performance of large language models on modern GPUs. ([Original Repo](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention))

[![arXiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

## Key Features

*   **Optimized NSA Implementation:** FSA provides a highly efficient, Triton-based implementation for the NSA selected attention module, especially beneficial for models with GQA group sizes less than 8.
*   **Significant Performance Gains:** Achieves speedups by reducing kernel-level memory access volume and computations.
*   **Wide Compatibility:** Tested and optimized for NVIDIA Ampere and Hopper GPUs, supporting fp16/bf16 data types, various GQA group sizes, and training/inference (prefill).
*   **Easy Integration:** Provides a ready-to-use `FlashSparseAttention` module, making it simple to integrate FSA into existing LLM pipelines.
*   **Comprehensive Benchmarking:** Includes scripts for detailed performance, correctness, and memory usage comparisons.

## Key Improvements

*   **Faster Kernel Execution:** FSA uses a novel kernel loop order to reduce unnecessary memory access and computations for padded data.
*   **Compatibility:** Supports a range of GPUs, datatypes, head sizes, and GQA sizes
*   **Lower Memory Access:** Optimizes memory access patterns to significantly reduce the overall memory access volume.

## News

*   **[Upcoming]**: Online profiling module for seamless NSA and FSA transitions.
*   **[2025-08]**: Published on Arxiv: [https://www.arxiv.org/abs/2508.18224](https://www.arxiv.org/abs/2508.18224)
*   **[2025-08]**: Beta version of one-step decoding released in [`fsa_preview`](fsa_preview).
*   **[2025-08]**: Open-sourced Flash-Sparse-Attention

## Method

FSA optimizes the NSA selected attention module by exchanging kernel loop order to batch query tokens that attend to the same KV block and store partial results. This approach reduces unnecessary memory access and computations for padded data.

## Advantages

FSA achieves significant speedups by lowering kernel-level memory access and computations.
<!-- Example: -->
<!-- <img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" /> -->

## Installation

**Dependencies:**

*   PyTorch >= 2.4
*   Triton >= 3.0
*   transformers >= 4.45.0
*   datasets >= 3.3.0
*   accelerate >= 1.9.0
*   flash-attn == 2.6.3

```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

```python
import torch
from fsa.module.fsa import FlashSparseAttention, RopeConfig

FSA = (
    FlashSparseAttention(...) # Configuration example from original README
    .cuda()
    .to(torch.bfloat16)
)
# ... rest of the usage example from original README ...
```

### Train with FSA

Training with FSA involves replacing the attention module. The primary step is to instantiate the FSA module and compute `cu_seqlens`. See the example in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Run detailed benchmarks using scripts in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh).

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA module using scripts in [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh).

> [!Tip]
> Use varied gqa, seqlen, block_size, topk arguments in benchmarking scripts for more comprehensive results.

## Performance

### Kernel Performance

<!-- Insert Kernel Performance image -->
<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

<!-- Insert End-to-end Performance image -->
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