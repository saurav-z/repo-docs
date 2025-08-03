# Lightning Thunder: Supercharge Your PyTorch Models ⚡

Lightning Thunder is a source-to-source compiler that accelerates your PyTorch models, making them run faster and more efficiently. [Explore the GitHub Repository](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Thunder Logo (Light Mode)" width="400px" style="max-width: 100%;">
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Thunder Logo (Dark Mode)" width="400px" style="max-width: 100%;">
</div>

**Key Features:**

*   🚀 **Significant Speedups:** Achieve up to 81% faster training and inference on various hardware.
*   🛠️ **Extensive Optimization:** Utilize custom kernels, fusion, quantization, and distributed strategies.
*   🧠 **User-Friendly:** Easily integrate plugins for out-of-the-box performance improvements.
*   ⚙️ **Composable Transformations:** Build custom optimizations with an ergonomic framework for performance experts.
*   🔥 **Hardware Ready:** Optimized for latest generation hardware, including NVIDIA Blackwell.
*   ⚙️ **Advanced Precision:** Support for FP4/FP6/FP8 and mixed precision training.
*   🧑‍💻 **Flexible Training & Inference:** Train and inference recipes, including support for LLMs and non-LLMs.
*   🧩 **Plugin Ecosystem:** Expand functionality with plugins for CUDA Graphs, distributed training (TP/PP/DP) and more.
*   💻 **Custom Kernels:** Leverage custom Triton kernels for even greater control.

**Benefits:**

*   **Faster Training:** Reduce training time and accelerate model development.
*   **Improved Inference:** Deploy models with lower latency and increased throughput.
*   **Resource Optimization:** Maximize hardware utilization for optimal performance.
*   **Ease of Use:** Simple integration with existing PyTorch code.

<div align="center">
  <pre>
  ✅ Up to 81% Faster Performance       ✅ FP4/FP6/FP8 Precision      ✅ Distributed Training
  ✅ Training & Inference Recipes      ✅ CUDA Graphs                  ✅ Custom Triton Kernels
  ✅ LLMs & Non-LLMs Support         ✅ Ready for NVIDIA Blackwell  ✅ Quantization & Fusion
  </pre>
</div>

<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml">
    <img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push" alt="CI testing">
  </a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml">
    <img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push" alt="General checks">
  </a>
  <a href="https://lightning-thunder.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/lightning-thunder/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main">
    <img src="https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg" alt="pre-commit.ci status">
  </a>
</div>

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick Start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance Benchmarks" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in a few simple steps:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

<details>
  <summary>Advanced Installation Options</summary>

### Blackwell Support

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

Optimize your PyTorch model with Thunder in a few lines:

```python
import torch
import torch.nn as nn
import thunder

# Define a model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Create input data
x = torch.randn(64, 2048)

# Run the model
y = thunder_model(x)

# Verify the results
torch.testing.assert_close(y, model(x))
```

## Examples

Explore how to use Lightning Thunder with various models and tasks:

*   **Speed up LLM training:** using LitGPT.
*   **Speed up Hugging Face BERT inference:** for faster NLP tasks.
*   **Speed up Hugging Face DeepSeek R1 distill inference:** for improved inference performance.
*   **Speed up Vision Transformer inference:** accelerate image processing tasks.
*   **Benchmarking HF models:** evaluate performance gains across different configurations.

```bash
# Run the benchmarks
python examples/quickstart/hf_llm.py
```

### Benchmarks

On a L4 machine from [Lightning Studio](https://lightning.ai):

```bash
Eager: 2273.22ms
Thunder: 1254.39ms
```

An 81% speedup!

## Plugins

Extend Thunder's functionality with plugins to apply optimizations like distributed training, quantization, CUDA Graphs, and more. Use plugins to easily swap optimizations for the best results on your setup.

-   Scale up with distributed strategies with DDP, FSDP, TP ()
-   Optimize numerical precision with FP8, MXFP8
-   Save memory with quantization
-   Reduce latency with CUDAGraphs
-   Debugging and profiling

Example: Reduce CPU overheads with CUDAGraphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Lightning Thunder accelerates your models through these stages:

1.  **Acquisition:** Interpret Python bytecode to create a straight-line Python program.
2.  **Transformation:** Apply optimizations such as distribution and precision changes.
3.  **Execution:** Route parts of the trace for optimal execution using fusion, specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
  <img src="docs/source/_static/images/how_it_works.png" alt="How Thunder Works" width="800px" style="max-width: 100%;">
</div>

## Performance

See the performance gains on a pre-training task using LitGPT on H100 and B200 hardware:

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Pre-training Performance" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is an open-source project.

💬 [Join the Discord](https://discord.com/invite/XncpTy7DSt)
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)