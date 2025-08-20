<!--
  SPDX-FileCopyrightText: 2023, Lightning AI, Inc.
  SPDX-License-Identifier: Apache-2.0
-->

<div align="center">
<a href="https://github.com/Lightning-AI/lightning-thunder">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</a>
</div>

<div align="center">
  <br/>
  ⚡️ **Supercharge your PyTorch models with Lightning Thunder, accelerating performance and simplifying optimization.**
  <br/>
  <br/>
</div>

## Key Features

*   🚀 **Significant Speedups:** Achieve up to 81% faster training and inference with optimized kernels and strategies.
*   🛠️ **Easy Optimization:** Apply custom kernels, fusions, quantization, and distributed strategies with ease.
*   🔌 **Composable Transformations:** Utilize plugins for model speed-ups out of the box, for optimal utilization of last generation hardware.
*   💡 **Understandable and Extensible:** Optimize AI models with composable transformations.
*   🧠 **FP4/FP6/FP8 Precision Support:** Optimize numerical precision.
*   🌐 **Distributed Training:** Supports TP/PP/DP distributed training strategies.
*   ⚙️ **Advanced Hardware Support:** Ready for NVIDIA Blackwell and other cutting-edge hardware.
*   🎯 **Versatile Application:** Accelerate LLMs, non-LLMs, and more.

<div align="center">

  ✅ Run PyTorch 40% faster &nbsp; &nbsp; ✅ Quantization &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ✅ Kernel fusion
  <br/>
  ✅ Training recipes &nbsp; &nbsp;  ✅ FP4/FP6/FP8 precision  &nbsp; &nbsp; ✅ Distributed TP/PP/DP
  <br/>
  ✅ Inference recipes  &nbsp;  ✅ Ready for NVIDIA Blackwell &nbsp; ✅ CUDA Graphs
  <br/>
  ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels &nbsp; &nbsp;  ✅ Compose all the above
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
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a> •
     <a target="_blank" href="https://github.com/Lightning-AI/lightning-thunder" style="margin: 0 10px;">GitHub</a>
  </div>
</div>

&#160;

## Quick Start

Install Thunder with a single command:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

<details>
  <summary>Advanced Installation Options</summary>

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

Optimize your PyTorch model with just a few lines of code:

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

### Speed Up LLM Training

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

### Speed up Hugging Face BERT Inference

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

### Speed up Hugging Face DeepSeek R1 distill inference

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

Run the example script for a speed comparison:

```bash
python examples/quickstart/hf_llm.py
```

On an L4 machine from [Lightning AI](https://lightning.ai), you can expect:

```bash
Eager: 2273.22ms
Thunder: 1254.39ms
```

81% faster 🏎️! Quite the speedup ⚡️

### Speed up Vision Transformer Inference

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

### Benchmarking HF Models

See the `examples/quickstart/hf_benchmarks.py` script for benchmarking.  Here are the speedups you can expect:

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

Thunder plugins provide easy access to powerful optimizations like:

-   Distributed training strategies (DDP, FSDP, TP)
-   FP8, MXFP8 numerical precision optimization
-   Memory savings via quantization
-   Latency reduction using CUDAGraphs
-   Debugging and profiling

Apply plugins by using the `plugins=` argument in `thunder.compile`:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Thunder optimizes PyTorch models in three stages:

1.  ⚡️ **Acquisition:** Interprets Python bytecode and produces a straight-line Python program.
2.  ⚡️ **Transformation:** Modifies the computation trace for distribution and precision changes.
3.  ⚡️ **Execution Routing:** Executes the trace using:
    *   Fusion (`NVFuser`, `torch.compile`)
    *   Specialized libraries (e.g., `cuDNN SDPA`, `TransformerEngine`)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

The acquired trace for a simple MLP looks like this:

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

Thunder delivers significant performance gains.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Thunder is a collaborative open-source project with significant contributions from NVIDIA.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)