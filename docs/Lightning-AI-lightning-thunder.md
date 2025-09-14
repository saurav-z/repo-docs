<!-- Improved & Summarized README with SEO Optimization -->

<div align="center">

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

</div>

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>

**Lightning Thunder empowers you to optimize and accelerate your PyTorch models with ease.**

</div>

---

## Key Features:

*   ⚡ **Blazing Fast Performance:** Achieve significant speedups (up to 40% faster!) for your PyTorch models.
*   🧠 **Easy Optimization:** Simplify model optimization with custom kernels, fusions, quantization, and more.
*   🛠️ **Composable Transformations:**  Build and modify AI models through flexible, composable transformations, perfect for performance experts.
*   ⚙️ **Ready-to-Use Plugins:** Leverage pre-built plugins for instant model speed-ups, including distributed training, quantization, and CUDA graph integration.
*   🧮 **Precision Control:** Optimize numerical precision with FP4/FP6/FP8 support, and fine-tune your models for maximum efficiency.
*   🌐 **Distributed Training & Inference:**  Scale your models with distributed strategies (TP/PP/DP).
*   🚀 **Blackwell & CUDA Compatibility:** Fully compatible with the latest NVIDIA hardware and CUDA versions.
*   💡 **Custom Kernels:** Easily integrate and utilize custom Triton kernels for bespoke performance gains.
*   ✅ **Comprehensive Support:** Works with a wide range of models, including LLMs, non-LLMs, and vision models.

---

## Quick Start

Install Lightning Thunder:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

[See more installation options and advanced features in the documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

### Simple Example:

```python
import torch
import torch.nn as nn
import thunder

# Define a simple model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Create input and run the model
x = torch.randn(64, 2048)
y = thunder_model(x)

# Verify the output
torch.testing.assert_close(y, model(x))
```

## Examples

*   **Speed Up LLM Training:** Integrate Thunder with LitGPT for faster large language model training.
*   **Accelerate Hugging Face BERT Inference:** Optimize BERT models with Thunder for faster inference.
*   **Optimize Hugging Face DeepSeek R1 Distill Inference:**  Get significant speedups with DeepSeek models.
*   **Boost Vision Transformer Inference:**  Enhance the performance of vision models.
*   **Benchmarking HF Models:** Benchmarking script can be used to measure speedups for text generation, forward pass, forward pass with loss, and a full forward + backward computation.

### Sample Results

*   **DeepSeek-ai/DeepSeek-R1-Distill-Llama-1.5B Model (H100 with torch=2.7.0):**
    *   Text generation: Up to 3.42x faster.
    *   Forward pass: Up to 1.63x faster.
    *   Forward pass + loss: Up to 1.64x faster.
    *   Forward + backward: Up to 1.69x faster.

## Plugins

Enhance your models with Thunder's powerful plugins:

*   **Distributed Training:** Scale models using DDP, FSDP, and TP.
*   **Precision Optimization:**  Use FP8, MXFP8 to improve efficiency.
*   **Quantization:**  Reduce memory footprint.
*   **CUDA Graphs:** Minimize CPU overhead.
*   **Debugging and Profiling:**  Tools for in-depth analysis.

To activate a plugin:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead") # Example: Enable CUDA Graphs
```

## How it Works

Thunder operates in three key stages:

1.  **Acquisition:** Interprets Python bytecode to produce a straight-line program.
2.  **Transformation:** Transforms the computation trace to make it distributed and adjust precision.
3.  **Execution:** Routes trace parts for execution with fusion, specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

---

## Performance

[See the performance gains on pre-training tasks with LitGPT](docs/source/_static/images/pretrain_perf.png) using H100 and B200 hardware!

## Community

Join the Lightning Thunder community:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)
*   [View the source code on GitHub](https://github.com/Lightning-AI/lightning-thunder)