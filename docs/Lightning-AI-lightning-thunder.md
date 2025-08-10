# Lightning Thunder: Supercharge Your PyTorch Models with Blazing Speed ⚡

**Accelerate your PyTorch models and unlock their full potential with Lightning Thunder, a source-to-source compiler for PyTorch, designed for speed, understanding, and extensibility.**  [Explore the original repo](https://github.com/Lightning-AI/lightning-thunder).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

**Key Features:**

*   ✅ **Up to 40% Faster PyTorch**: Achieve significant speedups out-of-the-box.
*   ✅ **Quantization Support**: Utilize FP4/FP6/FP8 precision for memory efficiency and faster inference.
*   ✅ **Kernel Fusion**: Automatic optimization for efficient computation.
*   ✅ **Distributed Training**: Utilize TP/PP/DP strategies.
*   ✅ **CUDA Graphs**: Reduce CPU overhead for improved performance.
*   ✅ **Custom Kernel Integration**: Extend functionality with Triton kernels.
*   ✅ **LLMs and Beyond**:  Optimized for various models, including LLMs and non-LLMs.
*   ✅ **NVIDIA Blackwell Ready**: Optimized for latest hardware.
*   ✅ **Composable Transformations**: Build and customize your model optimizations with ease.

<div align='center'>

<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>

</div>

<div align="center">
<img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Thunder in a few simple steps:

1.  **Install Thunder:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    **Advanced Install Options**  (Blackwell, additional executors, bleeding edge, development) are detailed in the original README.

2.  **Import and Compile Your Model:**

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

## Examples

Explore practical use cases and performance improvements:

*   **Speed up LLM training:**  (using LitGPT)
    *   Follow the instructions in the original README.
*   **Speed up Hugging Face BERT inference:**
    *   Follow the instructions in the original README.
*   **Speed up Hugging Face DeepSeek R1 distill inference:**
    *   Follow the instructions in the original README.
*   **Speed up Vision Transformer Inference:**
    *   Follow the instructions in the original README.
*   **Benchmarking HF models:**
    *   The script `examples/quickstart/hf_benchmarks.py` demonstrates how to benchmark a model for text generation, forward pass, forward pass with loss, and a full forward + backward computation.

## Plugins

Enhance your model optimization with Thunder's plugin system:

*   **Distributed Training:** Scale your model with DDP, FSDP, and TP.
*   **Precision Tuning**: Optimize numerical precision with FP8 and MXFP8.
*   **Quantization:**  Reduce memory footprint and accelerate inference.
*   **CUDA Graphs**:  Reduce CPU overhead for improved performance.
*   **Debugging and Profiling:** Tools for analyzing and understanding your model.

    Example:  Reduce CPU overhead via CUDAGraphs
    ```python
    thunder_model = thunder.compile(model, plugins="reduce-overhead")
    ```

## How It Works

Thunder's architecture is based on three stages:

1.  **Acquisition**: Interpret Python bytecode and produce a straight-line Python program.
2.  **Transformation**:  Optimize the computation trace for various improvements.
3.  **Execution**: Route parts of the trace using fusion (NVFuser, torch.compile), specialized libraries, custom kernels, and eager PyTorch operations.

## Community

Join the Lightning Thunder community:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)