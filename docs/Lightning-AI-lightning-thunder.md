<!-- Improved README for Lightning Thunder -->

<div align="center">
  <h1>Lightning Thunder: Supercharge Your PyTorch Models ⚡</h1>
</div>

<div align="center">
    <img alt="Thunder Logo" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
    <img alt="Thunder Logo" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

<div align="center">
  <p><b>Lightning Thunder is a source-to-source compiler designed to dramatically accelerate your PyTorch models.</b></p>
</div>

<div align="center">
  <pre>
  ✅ Up to 40% faster PyTorch models   ✅ FP4/FP6/FP8 Precision       ✅ Distributed Training
  ✅ Quantization                     ✅ Kernel Fusion               ✅ CUDA Graphs
  ✅ Training & Inference Recipes     ✅ NVIDIA Blackwell Ready      ✅ LLMs, non-LLMs, and more
  </pre>
</div>

<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder">
    <img src="https://img.shields.io/github/stars/Lightning-AI/lightning-thunder?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/Lightning-AI/lightning-thunder">
    <img src="https://img.shields.io/github/license/Lightning-AI/lightning-thunder" alt="License">
  </a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml">
    <img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push" alt="CI Testing">
  </a>
  <a href="https://lightning-thunder.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/lightning-thunder/badge/?version=latest" alt="Documentation">
  </a>
</div>

<div align="center">
  <a href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Documentation</a>
  <a href="#quick-start" style="margin: 0 10px;">Quick Start</a>
  <a href="#examples" style="margin: 0 10px;">Examples</a>
  <a href="#performance" style="margin: 0 10px;">Performance</a>
</div>

<div align="center">
  <img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Key Features

*   **Accelerated PyTorch:** Achieve significant speedups without extensive code changes.
*   **Model Optimization:** Easily integrate custom kernels, fusions, quantization, and distributed strategies.
*   **Pre-Built Plugins:** Leverage ready-to-use plugins for CUDA graphs, mixed precision, and more.
*   **Performance Experts:** A powerful framework for understanding, modifying, and optimizing AI models through composable transformations.
*   **LLM and Non-LLM Support:** Ready to use with both large language models and other model architectures.
*   **Hardware Ready:** Optimized for modern hardware, including NVIDIA Blackwell.

## Quick Start

Get started with Lightning Thunder in a few simple steps:

1.  **Installation:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    For advanced installation options including Blackwell support and optional executors, refer to the original [README](https://github.com/Lightning-AI/lightning-thunder).

2.  **Example Usage:**

    ```python
    import torch
    import torch.nn as nn
    import thunder

    # Define your model
    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

    # Compile the model with Thunder
    thunder_model = thunder.compile(model)

    # Perform inference
    x = torch.randn(64, 2048)
    y = thunder_model(x)

    # Verify results
    torch.testing.assert_close(y, model(x))
    ```

## Examples

Lightning Thunder seamlessly integrates with popular libraries and model architectures. Here are a few examples:

*   **Speeding up LLM Training** (LitGPT): Includes quick start installation instructions.
*   **Accelerating Hugging Face BERT Inference:** Example code provided.
*   **Optimizing Hugging Face DeepSeek R1 Distill Inference:** Demonstrates usage with DeepSeek models.
*   **Speeding up Vision Transformer Inference:** Example code using torchvision models.
*   **Benchmarking HF Models:** Learn how to measure speedups.

For detailed instructions and code snippets, see the original [README](https://github.com/Lightning-AI/lightning-thunder)

## Plugins

Enhance your model with Thunder's built-in plugins, which provide advanced optimizations:

*   **Distributed Training:** Scale up models with DDP, FSDP, and TP strategies.
*   **Precision Optimization:** Utilize FP8 and MXFP8 precision for enhanced performance.
*   **Quantization:** Reduce memory footprint and improve inference speed.
*   **CUDA Graphs:** Minimize CPU overhead using CUDA Graphs.
*   **Debugging and Profiling:** Built-in support for debugging and profiling.

Enable CUDA graphs to reduce CPU overhead:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## Performance

Lightning Thunder delivers impressive performance improvements:

<div align="center">
  <img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

Thunder can achieve significant speedups on a pre-training task using LitGPT on H100 and B200 hardware, relative to PyTorch eager. See the [original README](https://github.com/Lightning-AI/lightning-thunder) for detailed performance benchmarks.

## How it Works

Thunder transforms PyTorch models through three key stages:

1.  **Acquisition:** Interprets Python bytecode to create a straight-line Python program.
2.  **Transformation:** Modifies the computation trace for distribution and precision adjustments.
3.  **Execution:** Routes parts of the trace for execution through fusion, specialized libraries, custom kernels, or PyTorch eager operations.

For a deeper dive into Thunder's inner workings, see the [original README](https://github.com/Lightning-AI/lightning-thunder).

## Community

Lightning Thunder is an open-source project developed in collaboration with the community, with significant contributions from NVIDIA.

*   **Discord:** Join the community for help and discussions: [Discord](https://discord.com/invite/XncpTy7DSt).
*   **License:** Apache 2.0 - See the [LICENSE](https://github.com/Lightning-AI/litserve/blob/main/LICENSE) file.

**[Visit the Lightning Thunder GitHub Repository for more details](https://github.com/Lightning-AI/lightning-thunder).**