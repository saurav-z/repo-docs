<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Lightning Thunder" width="400px" style="max-width: 100%;">
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Lightning Thunder" width="400px" style="max-width: 100%;">
</div>

# Lightning Thunder: Supercharge Your PyTorch Models with Ease

**Lightning Thunder accelerates your PyTorch models, delivering significant performance gains through advanced compilation and optimization techniques.**  Access the original repository on [GitHub](https://github.com/Lightning-AI/lightning-thunder).

**Key Features:**

*   ⚡ **Up to 81% Faster** model training and inference
*   ✅ **Easy to Use:**  Apply optimizations with simple plugins and the `thunder.compile()` function.
*   🧠 **Understandable:**  Gain insights into your model's execution with a transparent, Python-based intermediate representation.
*   ⚙️ **Extensible:**  Build custom kernels, fusions, and optimization strategies.
*   🚀 **Ready for Cutting-Edge Hardware:**  Supports FP4/FP6/FP8 precision, NVIDIA Blackwell, CUDA Graphs, and more.
*   🧩 **Composable Optimizations:** Combine quantization, kernel fusion, distributed training (TP/PP/DP), and custom Triton kernels.
*   🛠️ **Ready-to-Use Plugins:** Benefit from out-of-the-box speed-ups for both training and inference.
*   💻 **Supports LLMs and More:** Works seamlessly with Large Language Models (LLMs) and other PyTorch models.

<div align="center">
  <pre>
  ✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
  ✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
  ✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
  ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
  </pre>
</div>

<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml">
    <img alt="CI Testing" src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push">
  </a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml">
    <img alt="General Checks" src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push">
  </a>
  <a href="https://lightning-thunder.readthedocs.io/en/latest/?badge=latest">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/lightning-thunder/badge/?version=latest">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main">
    <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg">
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
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started by installing Thunder:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

*For more installation options, see the [Thunder documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).*

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

Here's how to quickly optimize your PyTorch models:

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

## Examples

*   **Speed up LLM Training**
    *   Install LitGPT: `pip install --no-deps 'litgpt[all]'`
    *   Run the provided example code (see original README).
*   **Speed up HuggingFace BERT Inference**
    *   Install Transformers: `pip install -U transformers`
    *   Run the example code (see original README).
*   **Speed up HuggingFace DeepSeek R1 Distill Inference**
    *   Install Transformers: `pip install -U transformers`
    *   Run the example code (see original README).
*   **Speed up Vision Transformer Inference**
    *   Run the example code (see original README).

To assess performance gains, execute the `examples/quickstart/hf_llm.py` script. The output will provide a direct comparison of eager execution versus Thunder execution.

### Benchmarking HF models

The script `examples/quickstart/hf_benchmarks.py` demonstrates how to benchmark a model for text generation, forward pass, forward pass with loss, and a full forward + backward computation.

On an H100 with torch=2.7.0 and nvfuser-cu126-torch27, running deepseek-ai/DeepSeek-R1-Distill-Llama-1.5B, the thunder executors (NVFuser and torch.compile) achieve the following speedups:

```
Text generation:
Thunder (nvfuser): 3.36× faster
Thunder (torch.compile): 3.42× faster

Forward pass:
Thunder (nvfuser): 1.51× faster
Thunder (torch.compile): 1.63× faster

Forward pass + loss:
Thunder (nvfuser): 1.55× faster
Thunder (torch.compile): 1.64× faster

Forward + backward:
Thunder (nvfuser): 1.51× faster
Thunder (torch.compile): 1.69× faster
```

## Plugins

Thunder plugins provide a modular way to apply optimizations.  Easily enable features such as:

*   Distributed strategies (DDP, FSDP, TP) for scaling.
*   Numerical precision optimization (FP8, MXFP8).
*   Memory savings through quantization.
*   Reduced latency with CUDAGraphs.
*   Debugging and Profiling

Enable a plugin, like CUDA Graphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder's compilation process involves three key stages:

1.  **Acquisition:** Thunder interprets Python bytecode to generate a straight-line Python program representation of your model.
2.  **Transformation:** The computation trace is transformed to apply optimizations, change precision, and more.
3.  **Execution:**  Parts of the trace are routed for optimized execution:

    *   Fusion (NVFuser, `torch.compile`)
    *   Specialized Libraries (e.g., `cuDNN SDPA`, `TransformerEngine`)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
  <img src="docs/source/_static/images/how_it_works.png" alt="How it Works" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder provides significant performance improvements. The graph below displays speed-ups obtained on a pre-training task using LitGPT on H100 and B200 hardware, relative to PyTorch eager execution.

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance" width="800px" style="max-width: 100%;">
</div>

## Community

Thunder is an open-source project built in collaboration with the community, with major contributions from NVIDIA.

*   💬 [Join the Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)