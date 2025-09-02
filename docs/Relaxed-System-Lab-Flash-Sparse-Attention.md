<!-- Improved & SEO-Optimized README -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" alt="Flash Sparse Attention" width="700">
  <br>
  <a href="https://arxiv.org/abs/2508.18224">
    <img src="https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square" alt="arXiv">
  </a>
</div>

**Flash Sparse Attention (FSA) offers a faster and more efficient approach to Native Sparse Attention (NSA), significantly boosting the performance of Large Language Models (LLMs) on modern GPUs.**

**[Visit the Original Repository](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)**

## Key Features & Benefits

*   🚀 **Optimized NSA Implementation:** FSA provides a highly optimized, Triton-based implementation of Native Sparse Attention for improved performance on modern NVIDIA GPUs.
*   💡 **Reduced Memory Access:** FSA's novel kernel design minimizes memory access and computations, leading to significant speedups.
*   ⚡ **Faster Training & Inference:** Achieve faster training and prefill latency for state-of-the-art LLMs with FSA.
*   🔄 **Versatile Compatibility:** Supports NVIDIA Ampere and Hopper GPUs, fp16/bf16 data types, varied GQA group sizes, and more.
*   📝 **Easy Integration:**  Simple to integrate into existing LLM architectures by replacing the attention module.

## Key Sections

### 📰 News

Stay updated on the latest developments, including:

*   🚀 **Upcoming:** Online profiling module for seamless NSA and FSA transitions.
*   💥 **August 2025:** Arxiv paper release.
*   🎈 **August 2025:** Beta version release of one-step decoding in the `fsa_preview` directory.
*   🎉 **August 2025:** Public release of Flash-Sparse-Attention, optimized for NSA.

### 💡 Method: How FSA Works

FSA improves the performance of NSA by re-arranging the kernel loop order. The main kernel batches query tokens that attend to the same key-value blocks, reducing unnecessary memory access and computations.  This approach avoids atomic operations during attention result accumulation.

*   **NSA vs. FSA Kernel Comparison:** (Image Included in Original README)

### ✅ Advantages: Speed and Efficiency

FSA achieves performance gains by reducing kernel-level memory access and computational overhead. Performance comparisons are shown in the original README.

### ✨ Features: Capabilities

FSA provides an efficient Triton-based implementation for the NSA selected attention module, particularly for GQA group sizes smaller than 8, which are common in cutting-edge LLMs. It falls back to the original NSA implementation for group sizes >= 8.

**Supported Features:**

*   **GPUs:** NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM)
*   **Data Types:** fp16 and bf16
*   **Head Dimension:** Same head dimension (<= 256) across query, key, and value
*   **GQA Group Sizes:** 1 to 16
*   **Training & Inference:** Prefill

### 🚀 Installation: Get Started

Install the required dependencies to run FSA:

*   PyTorch >= 2.4
*   Triton >= 3.0
*   transformers >= 4.45.0
*   datasets >= 3.3.0
*   accelerate >= 1.9.0
*   flash-attn == 2.6.3

```bash
pip install -r requirements.txt
```

### 💻 Usage: Implementing FSA

Instantiate the `FlashSparseAttention` module and compute `cu_seqlens` for use:

```python
# (Example code from original README)
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

Train with FSA by integrating the module in place of the attention module.

### 📊 Evaluation: Benchmarking Performance

Benchmark FSA using provided scripts for:

*   **FSA Module:**  Use `scripts/run_unit_test.sh`
*   **FSA Selected Attention Module:** Use `scripts/run_unit_test_sel_attn.sh`

Adjust `gqa`, `seqlen`, `block_size`, and `topk` for comprehensive results.

### 📈 Performance: Results

*   **Kernel Performance:** (Graphs included in original README). Comparison of FSA, NSA, and Full Attention kernels.
*   **End-to-end Performance:** (Graphs included in original README). Training and prefill latency comparisons across various models.

### 📝 Citation

```
@article{yan2025flashsparseattentionalternative,
  title={Flash Sparse Attention: More Efficient Natively Trainable Sparse Attention},
  author={Yan, Ran and Jiang, Youhe and Yuan, Binhang},
  journal={arXiv preprint arXiv:2508.18224},
  year={2025}
}
```

### 🙏 Acknowledgments

*   NSA Paper: [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
*   NSA Reference Implementation: [Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)