<!--
   SPDX-License-Identifier: Apache-2.0
-->

# Lightning Thunder: Supercharge Your PyTorch Models with Ease

**Lightning Thunder drastically accelerates your PyTorch models, making them faster, more efficient, and easier to optimize.**  ⚡️

[Get started with Lightning Thunder!](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Blazing Fast Performance:** Achieve up to 40% speedups on PyTorch models.
*   **Simplified Optimization:** Easily apply custom kernels, fusions, quantization, and distributed strategies.
*   **User-Friendly for All:**
    *   **End Users:** Benefit from out-of-the-box speed improvements through plugins.
    *   **Performance Experts:** A composable framework for understanding, modifying, and optimizing AI models.
*   **Comprehensive Optimization Techniques:**
    *   FP4/FP6/FP8 precision support
    *   Training and inference recipes
    *   Distributed training with TP/PP/DP
    *   CUDA Graphs for reduced overhead
    *   Integration with NVIDIA Blackwell hardware
    *   Custom Triton kernels for tailored performance
*   **Wide Compatibility:** Works seamlessly with LLMs, non-LLMs, and more.
*   **Easy to Use:** Simple `compile` function to get started.
*   **Open Source & Community Driven:** Actively developed with community contributions.

<div align="center">
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

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Install Lightning Thunder using pip:

```bash
pip install lightning-thunder
pip install -U torch torchvision
pip install nvfuser-cu128-torch28 nvidia-cudnn-frontend  # if NVIDIA GPU is present
```

<details>
  <summary>For older versions of <code>torch</code></summary>

<code>torch==2.7</code> + CUDA 12.8

```bash
pip install lightning-thunder

pip install torch==2.7.0 torchvision==0.22
pip install nvfuser-cu128-torch27 nvidia-cudnn-frontend  # if NVIDIA GPU is present
```

<code>torch==2.6</code> + CUDA 12.6

```bash
pip install lightning-thunder

pip install torch==2.6.0 torchvision==0.21
pip install nvfuser-cu126-torch26 nvidia-cudnn-frontend  # if NVIDIA GPU is present
```

<code>torch==2.5</code> + CUDA 12.4

```bash
pip install lightning-thunder

pip install torch==2.5.0 torchvision==0.20
pip install nvfuser-cu124-torch25 nvidia-cudnn-frontend  # if NVIDIA GPU is present
```

</details>

<details>
  <summary>Advanced install options</summary>

### Install optional executors

```bash
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

### Hello World Example

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

### Speed up LLM Training

Install LitGPT:

```bash
pip install --no-deps 'litgpt[all]'
```

Run:

```python
import thunder
import torch
import litgpt

with torch.device("cuda"):
    model = litgpt.GPT.from_name("Llama-3.2-1B").to(torch.bfloat16)

thunder_model = thunder.compile(model)
inp = torch.ones((1, 2048), device="cuda", dtype=torch.int64)
out = thunder_model(inp)
out.sum().backward()
```

### Speed up Hugging Face BERT Inference

Install Transformers:

```bash
pip install -U transformers
```

Run:

```python
import thunder
import torch
import transformers

model_name = "bert-large-uncased"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

with torch.device("cuda"):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model.requires_grad_(False)
    model.eval()

    inp = tokenizer(["Hello world!"], return_tensors="pt")

thunder_model = thunder.compile(model)

out = thunder_model(**inp)
print(out)
```

### Speed up HuggingFace DeepSeek R1 distill inference

Install Hugging Face Transformers:

```bash
pip install -U transformers
```

Run:

```python
import torch
import transformers
import thunder

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

with torch.device("cuda"):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model.requires_grad_(False)
    model.eval()

    inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt")

thunder_model = thunder.compile(model)

out = thunder_model.generate(
    **inp, do_sample=False, cache_implementation="static", max_new_tokens=100
)
print(out)
```

### Vision Transformer Inference

```python
import thunder
import torch
import torchvision as tv

with torch.device("cuda"):
    model = tv.models.vit_b_16()
    model.requires_grad_(False)
    model.eval()

    inp = torch.randn(128, 3, 224, 224)

out = model(inp)

thunder_model = thunder.compile(model)

out = thunder_model(inp)
```

### Benchmarking HF models
To get an idea of the speedups, just run

```bash
python examples/quickstart/hf_llm.py
```

Here what you get on a L4 machine from [Lightning Studio](https://lightning.ai):

```bash
Eager: 2273.22ms
Thunder: 1254.39ms
```

81% faster 🏎️! Quite the speedup ⚡️

## Plugins

Plugins allow you to easily add optimizations like parallelism and quantization.

Example: Reduce CPU overhead with CUDA Graphs

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Lightning Thunder accelerates your PyTorch models through a source-to-source compilation approach.

1.  **Acquisition:** Interprets Python bytecode to create a straight-line program.
2.  **Transformation:** Optimizes the computation trace for distribution and precision.
3.  **Execution:** Routes the optimized trace to various execution backends.

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Lightning Thunder delivers significant speedups, as demonstrated by pre-training LitGPT on H100 and B200 hardware.

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is an open-source project built with community collaboration, and contributions from NVIDIA.

💬 [Join our Discord](https://discord.com/invite/XncpTy7DSt)
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)