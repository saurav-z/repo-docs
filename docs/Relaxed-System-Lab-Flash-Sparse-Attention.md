<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention Title" width="6200" height="1294">
</div>

---

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

# Flash Sparse Attention (FSA): Accelerating Large Language Models with Efficient Sparse Attention

**Flash Sparse Attention (FSA) provides an optimized implementation for Native Sparse Attention (NSA), enabling faster and more efficient training and inference of large language models (LLMs) on modern NVIDIA GPUs.  [Learn more at the original repository](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention).**

**Key Features:**

*   **Optimized NSA Kernel:**  FSA implements a novel kernel design to enhance the performance of Native Sparse Attention.
*   **Improved Performance:** Achieve significant speedups compared to traditional NSA, especially for GQA group sizes less than 8.
*   **Compatibility:**  Tested on NVIDIA Ampere and Hopper GPUs, supporting fp16 and bf16 data types, and various GQA group sizes.
*   **Easy Integration:** FSA can be easily integrated into existing LLM training pipelines by replacing the attention module.
*   **Triton-Based:** Leverages the power of Triton for optimized kernel implementations.

**Table of Contents:**

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

FSA optimizes the Native Sparse Attention (NSA) selected attention module by re-arranging the kernel loop order.  Instead of looping over query tokens in the outer loop and KV blocks in the inner loop (as in the original NSA), FSA reverses this, looping over KV blocks in the outer loop and query tokens in the inner loop. This allows for a reduction in unnecessary memory access and padding computations.

FSA decouples the computation into three main kernels:

1.  **Main Kernel:** Batches query tokens that attend to the same KV block and stores partial results.
2.  **Reduction Kernel:** Accumulates attention results for each query token.
3.  **Online Softmax Kernel:** Handles online softmax statistics computation.

The core of FSA's performance gains lies in minimizing memory access and avoiding `atomic` additions, leading to more efficient computations.

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages

🚀 FSA accelerates performance by significantly reducing kernel-level memory access and computational overhead.

Comparison of execution latency under different GQA group sizes, block size $B_K=64$,  topk-k value $T=16$, 64K sequence length, and 4 KV heads, the execution latency of our method is normalized to 1:

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />

## Features

FSA provides an optimized Triton-based implementation for the NSA selected attention module, particularly beneficial for GQA group sizes less than 8, common in modern LLMs, on NVIDIA GPUs. It falls back to the original NSA implementation for GQA group sizes greater than or equal to 8.

FSA has been rigorously tested with the following:

*   NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)
*   fp16 and bf16 data types.
*   Head dimensions up to 256.
*   Varied GQA group sizes, ranging from 1 to 16.
*   Training and inference (prefill).

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

Use the `FlashSparseAttention` class (in `fsa/module/FSA.py`) as demonstrated below:

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

Internally, the code utilizes the `FSATopkSparseAttention` class (`fsa/ops/FSA_topk_sparse_attention.py`), which provides the optimized kernels for the NSA selected attention module.

### Train with FSA

Integrating FSA into a training pipeline is straightforward.  Simply replace the attention module with the FSA module and compute `cu_seqlens` accordingly.  See the example in `test/train.py` (`SparseLlamaAttention`) for an illustration.

## Evaluation

### Benchmark FSA Module

Run the unit tests for the FSA module with the following command, provided in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh):

*   These scripts provides correctness comparison of forward and backward outputs, performance comparison, and memory usage comparison.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module using the scripts found in [`scripts/run_unit_test_sel_attn.sh`].

> [!Tip]
> For comprehensive benchmarking, vary the `gqa`, `seqlen`, `block_size`, and `topk` arguments in the provided scripts. Benchmarking the FSA selected attention module typically yields a higher speedup.

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

*   NSA paper: [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   NSA reference implementation: [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)