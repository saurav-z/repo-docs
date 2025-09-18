# Flash Sparse Attention (FSA): Accelerating Sparse Attention for LLMs

**Flash Sparse Attention (FSA) offers a novel kernel design to efficiently implement Native Sparse Attention (NSA) for faster and more performant Large Language Models (LLMs) on modern GPUs.**  Find the original repository [here](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention).

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

## Key Features

*   **Optimized NSA Implementation:** FSA provides a high-performance, Triton-based implementation for the NSA selected attention module.
*   **Efficient for Small GQA Groups:** FSA excels with Grouped-Query Attention (GQA) group sizes smaller than 8, common in leading LLMs.
*   **Broad Hardware Compatibility:**  Supports NVIDIA Ampere and Hopper GPUs (e.g., A100, H100) and datatypes like fp16 and bf16.
*   **Flexible GQA Support:** Compatible with various GQA group sizes (1-16) and supports both training and inference (prefill).
*   **Comprehensive Benchmarking:** Includes scripts for benchmarking the FSA module and the optimized NSA selected attention module, providing performance and correctness comparisons.

## News

*   **[Upcoming]** 🚀 Online profiling module for seamless NSA and FSA transition.
*   **[2025-08]** 💥 Published [Arxiv paper](https://www.arxiv.org/abs/2508.18224).
*   **[2025-08]** 🎈 Released beta version of one-step decoding ([`fsa_preview`](fsa_preview)).
*   **[2025-08]** 🎉 Open-sourced `Flash-Sparse-Attention`, optimizing NSA.

## Method

FSA optimizes the NSA kernel by changing its loop order, improving memory access and reducing padding-related computations. Key features:

*   **Kernel Reordering:** FSA loops over KV blocks in the outer loop and query tokens in the inner loop.
*   **Three-Kernel Decomposition:** Separates computation into main, reduction, and online softmax kernels for efficiency.
*   **Reduced Memory Access:** Minimizes unnecessary memory access and computations for padded data.

## Advantages

*   **Significant Speedup:** FSA offers substantial performance gains by reducing kernel-level memory access and computations.
*   **Performance improvements:** FSA consistently outperforms original NSA and Full Attention across varied configurations as shown in the provided images.

## Installation

1.  **Requirements:**

    *   PyTorch >= 2.4
    *   Triton >= 3.0
    *   transformers >= 4.45.0
    *   datasets >= 3.3.0
    *   accelerate >= 1.9.0
    *   flash-attn == 2.6.3

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Instantiate FSA Module

The `FlashSparseAttention` class is provided for easy use.

```python
import torch
from fsa.module.fsa import FlashSparseAttention, RopeConfig

FSA = (
    FlashSparseAttention(...) # Module Initialization
    .cuda()
    .to(torch.bfloat16)
)
# Input and Forward Pass...
y = FSA(x, cu_seqlens)
loss = (y * torch.randn_like(y)).sum(-1).mean()
loss.backward()
```

### Train with FSA

Integrate FSA into your LLM training by replacing the attention module.

*   Instantiate the FSA module.
*   Compute `cu_seqlens`.
*   See `SparseLlamaAttention` ([``test/train.py``](test/train.py)) for an example.

## Evaluation

### Benchmark FSA Module

Run the unit tests in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh) for benchmarking.

### Benchmark FSA Selected Attention Module

Use the commands in [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh) for detailed NSA selected attention module benchmarking.

> [!Tip]
> Adjust `gqa`, `seqlen`, `block_size`, and `topk` in the scripts for comprehensive benchmarking.

## Performance

### Kernel Performance

> Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations.

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

*   NSA paper: [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   NSA reference implementation: [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)