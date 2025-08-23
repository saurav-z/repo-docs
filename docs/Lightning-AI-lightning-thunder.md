<!--
  _   _  _   _  _   _  _   _  _   _  _   _  _   _  _   _
 | \ | || \ | || \ | || \ | || \ | || \ | || \ | || \ | |
 |_/|_||_/|_||_/|_||_/|_||_/|_||_/|_||_/|_||_/|_|
  Copyright (c) 2024 Lightning AI. All rights reserved.
-->
<div align="center">

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

</div>

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>
**Lightning Thunder is a source-to-source compiler that accelerates PyTorch models, making them faster, more efficient, and easier to optimize.** [Explore the Lightning Thunder Repository](https://github.com/Lightning-AI/lightning-thunder)

</div>

---

Lightning Thunder empowers both **end-users** and **performance experts** to optimize PyTorch models with ease. Leverage out-of-the-box plugins for immediate speed improvements or dive deep into model transformations for advanced customization.

**Key Features:**

*   🚀 **Significant Speedups:** Achieve up to 81% faster training and inference with minimal code changes.
*   🛠️ **Customizable Optimization:** Easily integrate custom kernels, fusion techniques, quantization, and distributed strategies.
*   💡 **User-Friendly:** Simple integration via a `compile` function, enabling rapid acceleration of existing PyTorch models.
*   ⚙️ **Versatile Plugins:** Utilize built-in plugins for quantization, CUDA graphs, and distributed training (TP/PP/DP), or create your own.
*   🧠 **Performance-Focused:** Provides a framework for understanding, modifying, and optimizing AI models through composable transformations.
*   💻 **Wide Compatibility:** Supports LLMs, non-LLMs, and is optimized for the latest NVIDIA hardware, including Blackwell.

<div align='center'>

<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>

</div>

<div align="center">

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)

</div>

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

---

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in minutes!

Install Thunder via pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

<details>
  <summary>Advanced install options</summary>

### Blackwell support

For Blackwell you'll need CUDA 12.8

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com

pip install lightning-thunder
```

### Install additional executors

These are optional, feel free to mix and match

```bash
# cuDNN SDPA
pip install nvidia-cudnn-frontend

# Float8 support (this will compile from source, be patient)
pip install "transformer_engine[pytorch]"
```

### Install Thunder bleeding edge

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
```

### Install Thunder for development

```bash
git clone https://github.com/Lightning-AI/lightning-thunder.git
cd lightning-thunder
pip install -e .
```

</details>

### Hello World

Here's a basic example of how to use Lightning Thunder:

```python
import torch.nn as nn
import thunder
import torch

# Define your PyTorch model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Create input data
x = torch.randn(64, 2048)

# Run the model
y = thunder_model(x)

# Verify the output (optional)
torch.testing.assert_close(y, model(x))
```

## Examples

Explore Lightning Thunder's capabilities with these example use cases:

*   [Speed up LLM training](https://github.com/Lightning-AI/lightning-thunder#speed-up-llm-training)
*   [Speed up HuggingFace BERT inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-bert-inference)
*   [Speed up HuggingFace DeepSeek R1 distill inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-deepseek-r1-distill-inference)
*   [Speed up Vision Transformer inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-vision-transformer-inference)
*   [Benchmarking HF models](https://github.com/Lightning-AI/lightning-thunder#benchmarking-hf-models)

## Plugins

Lightning Thunder's plugin system allows you to easily apply various optimizations to your models.

*   **Distributed Training:** Scale up training with DDP, FSDP, and TP.
*   **Numerical Precision:** Optimize with FP8, MXFP8.
*   **Memory Optimization:** Save memory with quantization.
*   **Latency Reduction:** Reduce overhead with CUDA Graphs.
*   **Debugging and Profiling:** Leverage debugging and profiling tools.

Example: Reduce CPU overhead with CUDA Graphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Lightning Thunder transforms your PyTorch models in three key stages:

1.  **Acquisition:**  The Python bytecode is interpreted to produce a straight-line Python program (the trace).
2.  **Transformation:** The computation trace is transformed to implement optimizations like distribution and precision changes.
3.  **Execution:** Parts of the trace are routed for execution via:
    *   Fusion (NVFuser, torch.compile)
    *   Specialized Libraries (e.g., cuDNN SDPA, TransformerEngine)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Lightning Thunder delivers significant performance gains. Here's a comparison of training speeds using LitGPT on H100 and B200 hardware.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community and get involved:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)