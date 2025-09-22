---
title: "Lightning Thunder: Supercharge Your PyTorch Models with Lightning-Fast Performance"
description: "Lightning Thunder is a source-to-source compiler for PyTorch, enabling significant speedups, quantization, and more. Optimize your AI models easily!"
keywords: "PyTorch, model optimization, compiler, performance, AI, machine learning, quantization, CUDA, Triton, LLM"
---

# Lightning Thunder: Unleash the Power of Your PyTorch Models ⚡

[<img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Thunder" width="400px" style="max-width: 100%;">
<img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Thunder" width="400px" style="max-width: 100%;">](https://github.com/Lightning-AI/lightning-thunder)

**Lightning Thunder empowers you to optimize your PyTorch models for unprecedented speed and efficiency.** This source-to-source compiler seamlessly integrates with your existing PyTorch code, offering a suite of advanced features to accelerate training and inference.

**Key Features:**

*   🚀 **Significant Speedups:** Achieve up to 40% faster PyTorch model execution out of the box.
*   ⚙️ **Model Optimization Made Easy:** Leverage custom kernels, fusions, quantization, and various distributed strategies.
*   💡 **User-Friendly:** Benefit from ready-to-use plugins for instant performance gains and optimal hardware utilization.
*   🛠️ **Extensible for Experts:** Customize and optimize AI models with composable transformations.
*   🧠 **Broad Compatibility:** Supports LLMs, non-LLMs, and is ready for modern hardware like NVIDIA Blackwell.
*   🔬 **Advanced Techniques:** Includes FP4/FP6/FP8 precision, distributed training (TP/PP/DP), CUDA Graphs, and custom Triton kernels.
*   🧩 **Composable:** Combine all features for optimal performance.

<pre>
✅ Up to 40% Faster PyTorch  ✅ Quantization                ✅ Kernel Fusion        
✅ Training & Inference Recipes ✅ FP4/FP6/FP8 Precision       ✅ Distributed TP/PP/DP 
✅ Ready for NVIDIA Blackwell ✅ CUDA Graphs          
✅ LLMs, non LLMs & more     ✅ Custom Triton Kernels       ✅ Compose all the above
</pre>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick Start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

<div align="center">
<img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in minutes:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

<details>
  <summary>Advanced Install Options</summary>

#### Blackwell Support

For Blackwell you'll need CUDA 12.8

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com

pip install lightning-thunder
```

#### Install Additional Executors

```bash
# cuDNN SDPA
pip install nvidia-cudnn-frontend

# Float8 support (this will compile from source, be patient)
pip install "transformer_engine[pytorch]"
```

#### Install Thunder Bleeding Edge

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
```

#### Install Thunder for Development

```bash
git clone https://github.com/Lightning-AI/lightning-thunder.git
cd lightning-thunder
pip install -e .
```

</details>

### Hello World

Optimize your PyTorch models with just a few lines of code:

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

```bash
pip install --no-deps 'litgpt[all]'
```

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

### Speed up HuggingFace BERT Inference

```bash
pip install -U transformers
```

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

```bash
pip install -U transformers
```

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

### Speed up Vision Transformer inference

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

Thunder's plugins enable you to easily apply various optimizations.

*   **Reduce CPU overhead with CUDAGraphs:** `thunder_model = thunder.compile(model, plugins="reduce-overhead")`

## How it Works

Thunder works in three stages:

1.  ⚡️ **Acquire:** Interprets Python bytecode to create a straight-line Python program.
2.  ⚡️ **Transform:** Optimizes the computation trace, modifying precision, adding distribution, etc.
3.  ⚡️ **Execute:** Routes operations for execution using:
    *   Fusion (`NVFuser`, `torch.compile`)
    *   Specialized libraries (e.g. `cuDNN SDPA`, `TransformerEngine`)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Lightning Thunder delivers significant performance gains. Here's a comparison on a pre-training task using LitGPT on H100 and B200 hardware, relative to PyTorch eager.

<div align="center">
<img alt="Thunder Performance" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community:

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)