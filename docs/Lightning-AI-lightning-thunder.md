<!-- Improved README with SEO Optimization -->

<div align="center">
  <h1>Lightning Thunder: Supercharge Your PyTorch Models</h1>
  <p><b>Lightning Thunder</b> is a source-to-source compiler for PyTorch, accelerating your models with custom kernels, fusions, quantization, and more. <a href="https://github.com/Lightning-AI/lightning-thunder">Check out the original repo!</a></p>
</div>

<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Thunder Logo (Light Mode)" width="400px" style="max-width: 100%;">
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Thunder Logo (Dark Mode)" width="400px" style="max-width: 100%;">
  <br><br>
</div>

## Key Features & Benefits

*   **Accelerate PyTorch**: Achieve significant speedups (up to 40% faster) with minimal code changes.
*   **Model Optimization**: Leverage advanced techniques like quantization (FP4/FP6/FP8), kernel fusion, and CUDA Graphs.
*   **Ease of Use**: Comes with pre-built plugins for instant performance gains on various hardware.
*   **Performance Tuning**: A powerful framework for experts to understand, modify, and optimize AI models through composable transformations.
*   **Broad Compatibility**: Supports LLMs, non-LLMs, and various training/inference recipes.
*   **Hardware Ready**: Optimized for NVIDIA Blackwell and other cutting-edge hardware.

<div align="center">
  <pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
  </pre>
</div>

<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml" target="_blank"><img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push" alt="CI Testing"></a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml" target="_blank"><img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push" alt="General Checks"></a>
  <a href="https://lightning-thunder.readthedocs.io/en/latest/?badge=latest" target="_blank"><img src="https://readthedocs.org/projects/lightning-thunder/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main" target="_blank"><img src="https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg" alt="pre-commit.ci status"></a>
</div>

<div align="center">
  <a href="#quick-start" style="margin: 0 10px;">Quick start</a> •
  <a href="#examples" style="margin: 0 10px;">Examples</a> •
  <a href="#performance" style="margin: 0 10px;">Performance</a> •
  <a href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
</div>

<br>

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance Chart" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in a few simple steps:

**Installation**:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

<details>
  <summary>Advanced Installation Options</summary>

### Blackwell Support

For Blackwell, you'll need CUDA 12.8

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
pip install lightning-thunder
```

### Install additional executors (optional)

```bash
# cuDNN SDPA
pip install nvidia-cudnn-frontend

# Float8 support (this will compile from source, be patient)
pip install "transformer_engine[pytorch]"
```

### Bleeding Edge

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
```

### Development Installation

```bash
git clone https://github.com/Lightning-AI/lightning-thunder.git
cd lightning-thunder
pip install -e .
```

</details>

**Hello World Example:**

```python
import torch.nn as nn
import torch
import thunder

model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
thunder_model = thunder.compile(model)

x = torch.randn(64, 2048)
y = thunder_model(x)

torch.testing.assert_close(y, model(x))
```

## Examples

Explore how to use Lightning Thunder with various models:

### Speed up LLM training

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

### Speed up HuggingFace BERT inference

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

To measure the speedups, run:

```bash
python examples/quickstart/hf_llm.py
```

Example result on a L4 machine from [Lightning Studio](https://lightning.ai):

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

The script `examples/quickstart/hf_benchmarks.py` demonstrates how to benchmark a model.

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

Thunder plugins allow you to apply various optimizations.

*   **Distributed Strategies**: DDP, FSDP, TP
*   **Numerical Precision**: FP8, MXFP8
*   **Memory Optimization**: Quantization
*   **Latency Reduction**: CUDAGraphs
*   **Debugging & Profiling**

For example, reduce CPU overheads via CUDAGraphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder operates in three main stages:

1.  **Acquisition**: Interprets Python bytecode to produce a straight-line Python program.
2.  **Transformation**: Transforms the computation trace for distribution and precision changes.
3.  **Execution**: Routes the trace for execution through fusion, specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
  <img src="docs/source/_static/images/how_it_works.png" alt="How Thunder Works" width="800px" style="max-width: 100%;">
</div>

Here's an example of the trace for a simple MLP:

```python
import thunder
import torch.nn as nn

model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 256))

thunder_model = thunder.compile(model)
y = thunder_model(torch.randn(4, 1024))

print(thunder.last_traces(thunder_model)[-1])
```

The acquired trace looks like this:

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

Lightning Thunder delivers significant speedups, especially for pre-training tasks.  See the chart below for performance improvements on H100 and B200 hardware using LitGPT.

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance Chart" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is an open-source project, built in collaboration with the community.

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)