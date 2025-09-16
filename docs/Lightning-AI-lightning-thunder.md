<!-- Hero Section -->
<div align="center">

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

</div>

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>
</div>

## About Lightning Thunder

**Lightning Thunder is a source-to-source compiler that empowers you to optimize your PyTorch models with ease, accelerating performance, and unlocking new capabilities.** This framework gives both end-users and performance experts a powerful toolset for model optimization.

*   **End-Users:** Access immediate speed-ups through pre-built plugins, optimized for the latest hardware.
*   **Performance Experts:** A highly ergonomic framework to understand, modify, and optimize AI models through composable transformations.

<div align='center'>

### Key Features

*   ✅ **Up to 81% Faster Performance** 🚀
*   ✅ **Quantization** (FP4/FP6/FP8)
*   ✅ **Kernel Fusion**
*   ✅ **Training and Inference Recipes**
*   ✅ **Distributed Training** (TP/PP/DP)
*   ✅ **CUDA Graphs**
*   ✅ **Support for LLMs, Non-LLMs, and more**
*   ✅ **Custom Triton Kernels**
*   ✅ **Composable Optimizations**
*   ✅ **Ready for NVIDIA Blackwell**

</div>

<div align='center'>

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?version=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)

</div>

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick Start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

&#160;

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in just a few steps.  See the [installation guide](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for more options.

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

<details>
  <summary>Advanced Install Options</summary>

### Blackwell Support

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
pip install lightning-thunder
```

### Install Additional Executors (Optional)

```bash
pip install nvidia-cudnn-frontend
pip install "transformer_engine[pytorch]"  # For Float8 support
```

### Bleeding Edge Install

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
```

### Development Install

```bash
git clone https://github.com/Lightning-AI/lightning-thunder.git
cd lightning-thunder
pip install -e .
```
</details>

### Hello World Example

Optimize your models with Thunder by defining a PyTorch module and compiling it:

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

Explore various use cases to understand Lightning Thunder's capabilities:

*   **LLM Training and Inference:**  Speed up LLM training with LitGPT.
*   **Hugging Face Integration:** Accelerate BERT and DeepSeek R1 inference.
*   **Vision Transformer Optimization:** Boost the performance of vision models.

### Speed up LLM training with LitGPT
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

### Speed up HuggingFace BERT inference
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

Run the example script for speed comparisons:

```bash
python examples/quickstart/hf_llm.py
```

Example Speedup (L4 machine from Lightning Studio):

*   **Eager:** 2273.22ms
*   **Thunder:** 1254.39ms
*   **81% Faster!** 🏎️⚡️

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

The script `examples/quickstart/hf_benchmarks.py` benchmarks a model for text generation, forward pass, forward pass with loss, and a full forward + backward computation.

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

Thunder plugins allow you to apply various optimizations easily.

-   **Distributed strategies:**  Scale with DDP, FSDP, and TP.
-   **Numerical precision:** Optimize with FP8 and MXFP8.
-   **Memory Optimization:** Leverage quantization.
-   **Latency Reduction:** Use CUDA Graphs.
-   **Debugging and Profiling:** Improve model understanding.

Example: Reduce CPU overhead with CUDA Graphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How Lightning Thunder Works

Thunder transforms and executes your PyTorch models:

1.  **Acquire:** Interprets Python bytecode to produce a straight-line Python program.
2.  **Transform:** Optimizes the computation trace for distribution and precision.
3.  **Execute:** Routes the trace via fusion (NVFuser, `torch.compile`), specialized libraries (e.g., cuDNN SDPA, TransformerEngine), custom Triton and CUDA kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

This is the trace for a simple MLP:

```python
import thunder
import torch.nn as nn

model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 256))

thunder_model = thunder.compile(model)
y = thunder_model(torch.randn(4, 1024))

print(thunder.last_traces(thunder_model)[-1])
```

Acquired trace example:

```python
def computation(input, t_0_bias, t_0_weight, t_2_bias, t_2_weight):
# input: "cuda:0 f32[4, 1024]"
# t_0_bias: "cuda:0 f32[2048]"
# t_0_weight: "cuda:0 f32[2048, 1024]"
# t_2_bias: "cuda:0 f32[256]"
# t_2_weight: "cuda:0 f32[256, 2048]"
t3 = ltorch.linear(input, t_0_weight, t_0_bias) # t3: "cuda:0 f32[4, 2048]"
t6 = ltorch.relu(t3, False) # t6: "cuda:0 f32[4, 2048]"
t10 = ltorch.linear(t6, t_2_weight, t_2_bias) # t10: "cuda:0 f32[4, 256]"
return (t10,)
```

## Performance

Thunder delivers significant speed improvements, as shown below:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join our community and contribute to the development of Lightning Thunder!

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)

<!-- Add a link to the original repo here: -->
For more details, visit the [Lightning Thunder repository](https://github.com/Lightning-AI/lightning-thunder).