<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Thunder" width="400px" style="max-width: 100%;">
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Thunder" width="400px" style="max-width: 100%;">
</div>

## Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder** is a source-to-source compiler for PyTorch that accelerates model training and inference, making your AI models faster, more efficient, and easier to optimize. Dive in to the official [GitHub repo](https://github.com/Lightning-AI/lightning-thunder) to learn more.

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Thunder Performance" width="800px" style="max-width: 100%;">
</div>

**Key Features:**

*   🚀 **Significant Speedups:** Achieve up to 81% faster performance in LLM inference and substantial gains in training.
*   ⚙️ **Optimized for Modern Hardware:** Leverages NVIDIA's latest hardware, including Blackwell, and supports FP4/FP6/FP8 precision.
*   🧩 **Composable Transformations:** Build custom kernels, fusions, quantization, and distributed strategies for expert-level customization.
*   🎯 **Out-of-the-Box Plugins:** Benefit from built-in plugins for quantization, CUDA Graphs, and distributed training (TP/PP/DP).
*   💡 **Easy to Use:** Simple integration with PyTorch models and straightforward compilation process.
*   🛠️ **Extensible:** Designed for performance experts to understand, modify, and optimize AI models through composable transformations.

---

## Quick Start

Get started with Lightning Thunder in minutes:

1.  **Install:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    *   **Advanced Install Options:**  Explore installation options for Blackwell support, additional executors, and bleeding-edge versions in the original [README](https://github.com/Lightning-AI/lightning-thunder#quick-start).

2.  **Example Usage:**

    ```python
    import torch
    import torch.nn as nn
    import thunder

    # Define a simple model
    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

    # Compile the model with Thunder
    thunder_model = thunder.compile(model)

    # Run inference
    x = torch.randn(64, 2048)
    y = thunder_model(x)

    torch.testing.assert_close(y, model(x))
    ```

## Examples

*   **Speed up LLM Training:** Integrates with LitGPT for faster large language model training.
*   **Accelerate Hugging Face Inference:**  Boosts inference speed for models like BERT, and DeepSeek.
*   **Optimize Vision Transformers:** Improves performance for models like ViT.
*   **Benchmarking HF models:** Includes example scripts to run benchmarks.

## Plugins

Thunder offers powerful plugins to enhance performance:

*   **Distributed Strategies:**  Scale up with DDP, FSDP, and TP.
*   **Numerical Precision Optimization:** Utilize FP8 and MXFP8 for reduced memory footprint and faster computation.
*   **Quantization:** Employ quantization techniques for model compression and speed gains.
*   **CUDA Graphs:** Reduce CPU overhead for lower latency.
*   **Debugging and Profiling:**  Tools for in-depth analysis and performance tuning.

## How It Works

Lightning Thunder accelerates PyTorch models through these key stages:

1.  **Acquisition:** Interprets Python bytecode to produce a straight-line Python program.
2.  **Transformation:** Transforms the computation trace for distribution and precision changes.
3.  **Execution:** Routes parts of the trace to various executors:

    *   Fusion (NVFuser, `torch.compile`)
    *   Specialized Libraries (e.g., `cuDNN SDPA`, `TransformerEngine`)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

    <div align="center">
      <img src="docs/source/_static/images/how_it_works.png" alt="How Thunder Works" width="800px" style="max-width: 100%;">
    </div>

## Community

*   💬 Get help on [Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 License: [Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)