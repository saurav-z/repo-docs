# Lightning Thunder: Supercharge Your PyTorch Models with Ease

Lightning Thunder is a source-to-source compiler for PyTorch, empowering you to optimize your models for speed and efficiency.  **[Explore the Lightning Thunder repository](https://github.com/Lightning-AI/lightning-thunder) to unlock your models' full potential.**

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   ⚡ **Blazing Fast:** Achieve up to 81% faster performance in LLM inference and significant speedups on other tasks.
*   ✅ **Out-of-the-Box Optimizations:** Leverage pre-built plugins for quantization, kernel fusion, and distributed training.
*   🔬 **Extensible and Customizable:**  Easily create custom kernels and transformations to tailor optimizations.
*   ⚙️ **Comprehensive Support:** Includes support for FP4/FP6/FP8 precision, CUDA Graphs, NVIDIA Blackwell, and more.
*   🚀 **Ease of Use:** Simple `thunder.compile()` interface for seamless integration with your existing PyTorch code.
*   🛠️ **Performance Tuning:** Provides a composable framework for performance experts to understand, modify, and optimize AI models.

## Quick Start

Get started with Lightning Thunder in three steps:

1.  **Install the required packages:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
    For more installation options, including Blackwell support and advanced features, see the [original documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

2.  **Define your model:**

    ```python
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    ```

3.  **Compile and run:**

    ```python
    import thunder
    import torch
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

## Examples

*   **Speed up LLM training** using LitGPT with the provided example.
*   **Accelerate Hugging Face BERT inference** with a simple code snippet.
*   **Optimize Hugging Face DeepSeek R1 distill inference** for faster results.
*   **Boost Vision Transformer inference** with optimized execution.
*   **Benchmarking HF models** to assess the performance gains of Thunder

## Performance

Thunder delivers significant performance improvements:

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Plugins

Thunder features a plugin architecture for easy model optimization:

*   **Distributed Strategies:**  DDP, FSDP, TP
*   **Precision Optimization:** FP8, MXFP8
*   **Memory Optimization:** Quantization
*   **Reduced Latency:** CUDAGraphs
*   **Debugging and Profiling**

## How It Works

Thunder's process involves three key stages:

1.  **Acquisition:** Converts Python bytecode to a straight-line Python program.
2.  **Transformation:** Optimizes the computation trace for distribution and precision changes.
3.  **Execution:** Routes trace parts for fusion, specialized libraries, custom kernels, and eager PyTorch operations.

## Community

Join the Lightning Thunder community:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)