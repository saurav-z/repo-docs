<!-- Banner Image -->
<img width="6200" height="1294" alt="Flash Sparse Attention" src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" />

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

# Flash Sparse Attention (FSA): Accelerating Large Language Models with Optimized Sparse Attention

**Flash Sparse Attention (FSA) offers a novel kernel design to dramatically improve the efficiency of Native Sparse Attention (NSA), making it faster and more accessible for training and inference on modern GPUs.** Explore the cutting-edge techniques used to optimize the sparse attention mechanism, crucial for high-performance LLMs.

**[View the original repository on GitHub](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)**

**Key Features:**

*   **Optimized NSA Implementation:** FSA provides an efficient, Triton-based implementation for Native Sparse Attention, enhancing performance, especially for smaller GQA group sizes.
*   **Significant Speedup:** Achieve substantial performance gains by reducing kernel-level memory access and computations compared to traditional NSA.
*   **Compatibility:** FSA is well-tested and compatible with NVIDIA Ampere and Hopper GPUs, and supports `fp16` and `bf16` data types.
*   **Versatile GQA Support:** FSA supports various Grouped-Query Attention (GQA) group sizes, from 1 to 16, making it suitable for state-of-the-art LLMs.
*   **Easy Integration:**  Instantiate the `FlashSparseAttention` module and easily integrate FSA into your LLM training pipeline with minimal modifications.

## What's New

*   **[Upcoming]**: 🚀 Online profiling module for seamless transitions between NSA and FSA.
*   **[August 2025]**: 💥 [Arxiv paper](https://www.arxiv.org/abs/2508.18224) released.
*   **[August 2025]**: 🎈 Beta version of one-step decoding available in [`fsa_preview`](fsa_preview).
*   **[August 2025]**: 🎉 Open-sourced `Flash-Sparse-Attention`, optimized for NSA.

## Method: Unveiling the FSA Kernel Design

FSA optimizes the NSA selected attention module by exchanging the kernel loop order to boost performance. This innovative approach splits the computation into three main kernels:

1.  **Main Kernel:** Batches query tokens that attend to the same key-value block, storing partial results to a buffer.
2.  **Reduction Kernel:** Accumulates attention results for each query token.
3.  **Online Softmax Kernel:** Handles online softmax statistics computation.

This design dramatically reduces unnecessary memory access and computations related to padded data, avoiding `atomic` additions.

**Visual Comparison:** The computational process of NSA (left) and FSA main kernel (right) are compared in the following image.

<img width="8817" height="3669" alt="NSA_FSA_cmop" src="https://github.com/user-attachments/assets/12250042-3c5d-40f3-82c3-d0ca443c4c45" />

## Advantages of FSA

🚀 **Superior Performance:** FSA's key advantage is significantly reduced kernel-level memory access and computations.

<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce9b792fe" />

## Features and Compatibility

FSA provides an optimized kernel implementation for the NSA selected attention module.

**Key Capabilities:**

*   **Hardware Support:** NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)
*   **Data Types:** `fp16` and `bf16`
*   **Head Dimension:**  Head dimensions less than or equal to 256 are supported across query, key, and value.
*   **GQA Support:** Flexible GQA group sizes, ranging from 1 to 16.
*   **Use Cases:** Training and inference (prefill).

## Installation

Ensure you have the required dependencies:

*   [PyTorch](https://pytorch.org/) >= 2.4
*   [Triton](https://github.com/openai/triton) >=3.0
*   [transformers](https://github.com/huggingface/transformers) >=4.45.0
*   [datasets](https://github.com/huggingface/datasets) >=3.3.0
*   [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
*   [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

Install the dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

Use the provided `FlashSparseAttention` class:

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

Under the hood, the [``FSATopkSparseAttention``](fsa/ops/FSA_topk_sparse_attention.py) class is called, providing the optimized kernels that accelerate the NSA selected attention module.

### Training with FSA

Integrating FSA into your training process is straightforward. Simply replace the attention module and compute `cu_seqlens`. An example is available in [`SparseLlamaAttention`](test/train.py).

## Evaluation

### Benchmark FSA Module

Run detailed benchmarks using the commands in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh). Benchmarking includes:

*   Correctness comparison (forward and backward pass).
*   Performance comparison.
*   Memory usage comparison.

### Benchmark FSA Selected Attention Module

Benchmark the optimized NSA selected attention module using the commands in [`scripts/run_unit_test_sel_attn.sh`](scripts/run_unit_test_sel_attn.sh).

> [!Tip]
> Experiment with different `gqa`, `seqlen`, `block_size`, and `topk` arguments in the scripts to gain a comprehensive understanding of FSA's performance on your hardware.  Benchmarking the FSA selected attention module typically offers a higher speedup.

## Performance Results

### Kernel Performance

> Performance comparisons of Triton-based FSA, NSA, and Full Attention kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $B_K$ and top-k value $Topk$, respectively. For FSA and NSA, the execution latency includes compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-End Performance

> End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.

<img width="6165" height="3093" alt="e2e_githubpic" src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" />

## Citation

If you find this project useful, please cite:

```
@article{yan2025flashsparseattentionalternative,
  title={Flash Sparse Attention: More Efficient Natively Trainable Sparse Attention},
  author={Yan, Ran and Jiang, Youhe and Yuan, Binhang},
  journal={arXiv preprint arXiv:2508.18224},
  year={2025}
}
```

## Acknowledgments

*   **NSA Paper**: [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   **NSA Reference Implementation**: [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)