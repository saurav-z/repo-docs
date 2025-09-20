<!--  Lightning Thunder: Supercharge Your PyTorch Models  -->
<div align="center">

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

</div>

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>

**Lightning Thunder drastically accelerates PyTorch models with a source-to-source compiler.**

</div>

---

## Key Features

*   **Blazing Fast Performance:** Achieve up to 81% speedup with optimized execution.
*   **Effortless Optimization:** Easily integrate custom kernels, fusions, quantization, and distributed strategies.
*   **Model Agnostic:** Compatible with Large Language Models (LLMs), Vision Transformers, and more.
*   **End-User Plugins:** Benefit from out-of-the-box speed enhancements for optimal hardware utilization.
*   **Performance Expert Toolkit:** Provides a composable framework for understanding, modifying, and optimizing AI models.
*   **FP4/FP6/FP8 Precision:** Train and inference models with varying numerical precision for memory efficiency and speed.
*   **Distributed Training:** Utilizes strategies like TP/PP/DP for scaling training across multiple devices.
*   **CUDA Graphs:** Leverages CUDA Graphs to reduce CPU overhead and improve performance.

---

## Quick Start

Lightning Thunder is easy to install:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

**For detailed installation options, including support for Blackwell and other advanced configurations, please refer to the [Installation Documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).**

### Simple Example

Compile your PyTorch models with a single line of code to start optimizing them:

```python
import thunder
import torch.nn as nn
import torch

# Define a simple model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile your model with Thunder
thunder_model = thunder.compile(model)

# Example input
x = torch.randn(64, 2048)

# Run the model
y = thunder_model(x)

# Verify results
torch.testing.assert_close(y, model(x))
```

---

## Examples

Explore practical examples to see Lightning Thunder in action:

*   **Speed up LLM training:** Install LitGPT and see your model's training speed up.
*   **Accelerate Hugging Face BERT inference:** Compile your BERT models for faster inference.
*   **Boost Hugging Face DeepSeek R1 distill inference:**  Optimize your DeepSeek models for increased performance.
*   **Optimize Vision Transformer Inference:** Utilize Thunder to make Vision Transformers faster

For more examples, including benchmarks, see the `examples` directory in the [Lightning Thunder repository](https://github.com/Lightning-AI/lightning-thunder).

---

## Plugins

Plugins are an essential feature for model optimization with Thunder. Use plugins to quickly apply optimization strategies to your model.

Thunder plugins include:

*   **Distributed Strategies:** Scale up training with DDP, FSDP, TP
*   **Numerical Precision:** Optimize with FP8, MXFP8
*   **Quantization:** Reduce memory usage with quantization.
*   **CUDA Graphs:** Reduce latency with CUDAGraphs
*   **Debugging and profiling:** Plugins to help during development.

---

## How It Works

Lightning Thunder operates in three core stages to accelerate your PyTorch models:

1.  **Acquisition:** Interprets Python bytecode, creating a straight-line Python program.
2.  **Transformation:** Transforms the computation trace for distribution and precision adjustments.
3.  **Execution:** Routes parts of the trace for execution using fusion, specialized libraries, custom kernels, and PyTorch operations.

---

## Performance

Lightning Thunder delivers significant speedups.  For example, on pre-training tasks using LitGPT on H100 and B200 hardware, Thunder demonstrates substantial performance gains compared to PyTorch eager execution.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

---

## Community

Lightning Thunder is an open-source project driven by a collaborative community, with significant contributions from NVIDIA. Join us!

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)
*   [View the Lightning Thunder repository on GitHub](https://github.com/Lightning-AI/lightning-thunder)