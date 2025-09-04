# Supercharge Your PyTorch Models with Lightning Thunder ⚡

**Lightning Thunder is a cutting-edge source-to-source compiler for PyTorch, enabling blazing-fast performance, advanced optimizations, and unparalleled flexibility.** Discover how you can accelerate your AI models with ease by visiting the [original repo](https://github.com/Lightning-AI/lightning-thunder).

<!-- Images for Light/Dark Mode -->
<div align="center">
  <img alt="Thunder - Light Mode" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder - Dark Mode" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Key Features

*   🚀 **Significant Speedups:** Achieve up to 40% faster PyTorch model execution out-of-the-box.
*   ⚙️ **Optimized for Modern Hardware:** Leverage the latest hardware with built-in support for NVIDIA Blackwell, CUDA Graphs, and more.
*   💡 **Simplified Optimization:** Easily integrate custom kernels, fusions, quantization (FP4/FP6/FP8), and distributed training strategies (TP/PP/DP).
*   🧩 **Composable Transformations:** Benefit from a flexible framework for understanding, modifying, and optimizing AI models.
*   🧠 **Ready-to-Use Plugins:** Enhance your models with a variety of pre-built plugins for various optimizations.
*   💻 **Broad Compatibility:** Supports LLMs, non-LLMs, and custom Triton kernels.

## Quick Start

Get started with Lightning Thunder in just a few steps:

1.  **Install Thunder:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

2.  **Define Your Model:**  Create your PyTorch model, such as a `torch.nn.Sequential` or a custom module.

3.  **Compile with Thunder:**

    ```python
    import thunder
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

    For advanced installation options, including Blackwell support and additional executors, please see the [original README](https://github.com/Lightning-AI/lightning-thunder).

## Examples

*   **Speed up LLM training:**  Install LitGPT and run the example.
*   **Speed up Hugging Face BERT inference:** Install Hugging Face Transformers and run the example.
*   **Speed up Hugging Face DeepSeek R1 distill inference:** Install Hugging Face Transformers and run the example.
*   **Speed up Vision Transformer inference:** Run the Vision Transformer example.
*   **Benchmarking HF models:**  Run the benchmarking script to measure performance gains on various Hugging Face models.

## Plugins

Thunder's plugin system allows you to easily apply various optimizations. Examples include:

*   Distributed strategies (DDP, FSDP, TP)
*   Numerical precision optimization (FP8, MXFP8)
*   Memory savings via quantization
*   Reduced latency with CUDAGraphs
*   Debugging and profiling

Activate plugins through the `plugins=` argument of `thunder.compile`.  For example:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Thunder's process is comprised of three primary stages:

1.  **Acquisition:**  Interprets Python bytecode to create a straight-line Python program.
2.  **Transformation:** Modifies the computation trace for distribution and precision changes.
3.  **Execution:** Routes parts of the trace for execution, leveraging fusion, specialized libraries, custom kernels, and PyTorch eager operations.

## Community

Join the Lightning Thunder community for support and collaboration:

*   💬 [Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)