<!-- Hero Section -->
<div align="center">
  <h1>Lightning Thunder: Supercharge Your PyTorch Models</h1>
  <p>Lightning Thunder dramatically accelerates PyTorch models through a source-to-source compiler, enabling advanced optimizations like quantization, kernel fusion, and distributed training.</p>
</div>

<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder">
    <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Thunder Logo - Light Mode" width="400px" style="max-width: 100%;">
    <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Thunder Logo - Dark Mode" width="400px" style="max-width: 100%;">
  </a>
  <br/><br/>
</div>

<!-- Overview -->
<div align="center">
  <p><strong>Unlock significant performance gains for your PyTorch models with Thunder, the source-to-source compiler.</strong></p>
</div>

<!-- Key Features -->
## Key Features

*   ⚡ **Significant Speedups:** Achieve up to 40% faster model execution and faster training.
*   🚀 **Model Optimization:** Automate advanced optimizations including:
    *   Kernel Fusion
    *   Quantization (FP4/FP6/FP8)
    *   Distributed Training (TP/PP/DP)
    *   CUDA Graphs
    *   Custom Triton Kernels
*   🛠️ **Extensible Architecture:** Easily integrate custom kernels and composable transformations.
*   🎯 **User-Friendly:**  Benefit from out-of-the-box performance improvements with easy-to-use plugins.
*   ⚙️ **Hardware-Aware:** Designed for optimal performance on modern NVIDIA hardware, including Blackwell.
*   🤖 **LLMs and Beyond:** Supports a wide range of models, including LLMs and other deep learning architectures.

<!-- Badges -->
<div align="center">
  <p>
    <a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
    </a>
    <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml">
      <img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push" alt="CI Testing">
    </a>
    <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml">
      <img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push" alt="General Checks">
    </a>
    <a href="https://lightning-thunder.readthedocs.io/en/latest/?badge=latest">
      <img src="https://readthedocs.org/projects/lightning-thunder/badge/?version=latest" alt="Documentation Status">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main">
      <img src="https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg" alt="pre-commit.ci status">
    </a>
  </p>
</div>

<!-- Navigation -->
<div align="center">
  <p>
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick Start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </p>
</div>

<!-- Get Started (Optional - Remove if not desired) -->
<!--
<div align="center">
  <a target="_blank" href="https://lightning.ai/docs/thunder/home/get-started">
    <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
  </a>
</div>
-->

<div align="center">
<img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

### Installation

Install Lightning Thunder with pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

<details>
  <summary>Advanced Installation Options</summary>

**Blackwell Support**

For Blackwell, you'll need CUDA 12.8.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
pip install lightning-thunder
```

**Install Additional Executors**

These are optional and can be combined.

```bash
# cuDNN SDPA
pip install nvidia-cudnn-frontend

# Float8 support (compiles from source)
pip install "transformer_engine[pytorch]"
```

**Bleeding Edge Install**

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
```

**Development Install**

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

# Define your model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile your model with Thunder
thunder_model = thunder.compile(model)

# Create input and perform inference
x = torch.randn(64, 2048)
y = thunder_model(x)

# Verify results (optional)
torch.testing.assert_close(y, model(x))
```

## Examples

Explore how Thunder can accelerate different model types:

### Speed up LLM Training

```python
import thunder
import torch
import litgpt
# ... (rest of the code as in the original README)
```

### Speed up HuggingFace BERT Inference

```python
import thunder
import torch
import transformers
# ... (rest of the code as in the original README)
```

### Speed up HuggingFace DeepSeek R1 distill inference

```python
import torch
import transformers
import thunder
# ... (rest of the code as in the original README)
```

### Speed up Vision Transformer Inference

```python
import thunder
import torch
import torchvision as tv
# ... (rest of the code as in the original README)
```

### Benchmarking HF Models

Run the script `examples/quickstart/hf_benchmarks.py` to benchmark models for text generation, forward pass, forward pass with loss, and full forward + backward computation.

Example speedups achieved on an H100 with torch=2.7.0 and nvfuser-cu126-torch27 using deepseek-ai/DeepSeek-R1-Distill-Llama-1.5B:

```
Text generation:
Thunder (nvfuser): 3.36× faster
Thunder (torch.compile): 3.42× faster
# ... (rest of the example results as in the original README)
```

## Plugins

Extend Lightning Thunder's capabilities with plugins for various optimizations.

*   **Distributed Training:** Scale up with DDP, FSDP, and TP.
*   **Precision Optimization:** Utilize FP8, MXFP8.
*   **Memory Savings:** Implement quantization techniques.
*   **Latency Reduction:** Leverage CUDAGraphs.
*   **Debugging and Profiling:** Analyze model behavior.

Reduce CPU overhead with CUDAGraphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Thunder uses a three-stage process to optimize your models:

1.  **Acquisition:**  Interprets Python bytecode to produce a straight-line program.
2.  **Transformation:** Transforms the computation trace for distribution and precision changes.
3.  **Execution:** Routes parts of the trace for optimized execution:
    *   Fusion (NVFuser, torch.compile)
    *   Specialized libraries (cuDNN SDPA, TransformerEngine)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
<img alt="How Thunder Works" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Lightning Thunder provides significant speedups, as demonstrated in the performance graph below for a pre-training task using LitGPT on H100 and B200 hardware.

<div align="center">
<img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community to get help and contribute:

*   💬 [Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)