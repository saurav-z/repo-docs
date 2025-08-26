# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder** is a source-to-source compiler that empowers you to dramatically accelerate your PyTorch models. Visit the [Lightning Thunder GitHub repo](https://github.com/Lightning-AI/lightning-thunder) to get started!

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Significant Speedups:** Achieve up to 2x faster training and inference on modern hardware.
*   **Ease of Use:** Simple integration with existing PyTorch models.
*   **Optimized Kernels:** Leverage cutting-edge techniques like kernel fusion, quantization (FP4/FP6/FP8), and CUDA Graphs.
*   **Flexible Plugins:** Extend functionality with composable transformations for distributed training (TP/PP/DP), numerical precision optimization, and more.
*   **Comprehensive Support:** Works with LLMs, non-LLMs, and is ready for NVIDIA Blackwell.

<div align='center'>

<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion        
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP 
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs          
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>

</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)

## Quick Start

Install Lightning Thunder and its dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

For detailed installation options, including Blackwell support and advanced configurations, see the [installation documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

### Example: Accelerating a Simple Model

```python
import torch
import torch.nn as nn
import thunder

# Define a simple model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Run the model
x = torch.randn(64, 2048)
y = thunder_model(x)

# Verify the output
torch.testing.assert_close(y, model(x))
```

## Examples

Explore the power of Lightning Thunder with these example use cases:

*   [Speed up LLM training](#speed-up-llm-training)
*   [Speed up HuggingFace BERT inference](#speed-up-huggingface-bert-inference)
*   [Speed up HuggingFace DeepSeek R1 distill inference](#speed-up-huggingface-deepseek-r1-distill-inference)
*   [Speed up Vision Transformer inference](#speed-up-vision-transformer-inference)
*   [Benchmarking HF models](#benchmarking-hf-models)

### Speed up LLM training

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

```bash
python examples/quickstart/hf_benchmarks.py
```

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

Enhance your model's performance with Thunder's flexible plugin system.  Plugins provide easy access to optimizations like:

*   Distributed training strategies (DDP, FSDP, TP)
*   Numerical precision optimization (FP8, MXFP8)
*   Memory savings with quantization
*   Reduced latency using CUDA Graphs

For example, to use CUDA Graphs for reduced CPU overhead:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Lightning Thunder compiles your PyTorch models in three key stages:

1.  **Acquisition:** Converts your model into a straight-line Python program by interpreting bytecode.
2.  **Transformation:** Optimizes the computation trace for performance, e.g.  by changing precision or distributing the computation.
3.  **Execution:** Runs optimized code using fusions (NVFuser, `torch.compile`), specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers significant speedups, as demonstrated by these results from a pre-training task with LitGPT:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is an open-source project developed with community collaboration and significant contributions from NVIDIA.

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)