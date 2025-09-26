# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder empowers you to accelerate your PyTorch models with ease, offering a suite of optimizations for peak performance.**  [Explore the original repo](https://github.com/Lightning-AI/lightning-thunder)

---

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Optimized Performance:** Achieve up to 40% faster PyTorch model execution.
*   **Model Optimization:**  Apply quantization, kernel fusion, and distributed strategies (TP/PP/DP).
*   **Precision Control:** Supports FP4/FP6/FP8 precision for efficient computation.
*   **Hardware-Ready:** Designed for NVIDIA Blackwell and CUDA Graphs.
*   **Flexible and Extensible:** Utilize custom Triton kernels and compose various optimizations.
*   **Built-in Recipes:** Ready-to-use training and inference recipes.
*   **Supports LLMs and More:** Works with a wide range of models, including LLMs and vision models.

---

## Why Choose Lightning Thunder?

*   **For End Users:** Benefit from out-of-the-box speed improvements with pre-built plugins.
*   **For Performance Experts:** Provides a powerful framework for understanding, modifying, and optimizing AI models.

---

<div align='center'>
<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>
</div>

---

## Quick Start

### Installation

Install Thunder using pip:

```bash
pip install lightning-thunder
```

Install dependencies:

```bash
pip install -U torch torchvision
pip install nvfuser-cu128-torch28 nvidia-cudnn-frontend  # if NVIDIA GPU is present
```

For older versions of torch, refer to the original README for compatibility and installation instructions.

### Basic Usage

Optimize your PyTorch models with a few lines of code:

```python
import torch.nn as nn
import thunder
import torch

model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
thunder_model = thunder.compile(model)
x = torch.randn(64, 2048)
y = thunder_model(x)
torch.testing.assert_close(y, model(x))
```

---

## Examples

*   **Speed Up LLM Training:** Easily integrate with LitGPT for faster training.
*   **Accelerate Hugging Face Inference:** Enhance inference speed for models like BERT and DeepSeek.
*   **Optimize Vision Transformers:** Improve the performance of models like ViT.

See the original README for detailed example code snippets and benchmark results.

---

## Plugins

Thunder plugins provide easy access to optimization techniques:

*   **Distributed Strategies:** DDP, FSDP, TP for scaling.
*   **Numerical Precision:** FP8, MXFP8 for efficient computation.
*   **Quantization:** Memory savings with quantization.
*   **CUDAGraphs:** Reduce CPU overhead and improve latency.
*   **Debugging and Profiling:** Tools for understanding your model's performance.

To use a plugin, simply include it in the `plugins=` argument of `thunder.compile`.  Example: `thunder.compile(model, plugins="reduce-overhead")`.

---

## How It Works

Thunder's three-stage process:

1.  **Acquisition:** Interprets Python bytecode to create a straight-line program.
2.  **Transformation:** Transforms the trace for distribution and precision changes.
3.  **Execution:** Routes parts of the trace for optimized execution using techniques like fusion, specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

---

## Performance

Thunder delivers significant speedups on various hardware, as illustrated by the performance charts in the original README.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

---

## Community

Thunder is an open-source project built in collaboration with the community and NVIDIA.

*   💬 [Join the Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)