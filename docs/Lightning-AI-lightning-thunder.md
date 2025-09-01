<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

Lightning Thunder is a source-to-source compiler that helps you optimize PyTorch models for faster training and inference, making them more efficient than ever. [Explore Lightning Thunder on GitHub](https://github.com/Lightning-AI/lightning-thunder).

## Key Features

*   **Up to 40% Faster PyTorch Models:** Achieve significant speedups out-of-the-box for both training and inference.
*   **Effortless Optimization:** Easily integrate custom kernels, fusions, quantization, and distributed strategies.
*   **Ready for Modern Hardware:** Designed for optimal performance on the latest NVIDIA hardware, including Blackwell.
*   **Versatile Application:** Accelerate a wide range of models, including LLMs, Vision Transformers, and more.
*   **Composable Transformations:** Offers an ergonomic framework for understanding, modifying, and optimizing AI models.
*   **Extensive Plugin Support:** Leverage pre-built plugins for distributed training (DDP, FSDP, TP), mixed-precision training (FP8, MXFP8), memory optimization, and CUDA graph integration.

## Getting Started

### Installation

Install Lightning Thunder and dependencies using pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

**Advanced Installation Options**

*   [Blackwell Support](#blackwell-support)
*   [Install Additional Executors](#install-additional-executors)
*   [Install Thunder Bleeding Edge](#install-thunder-bleeding-edge)
*   [Install Thunder for Development](#install-thunder-for-development)

### Quick Example

Optimize your PyTorch models with just a few lines of code:

```python
import torch.nn as nn
import thunder
import torch

# Define your model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile your model with Thunder
thunder_model = thunder.compile(model)

# Perform inference
x = torch.randn(64, 2048)
y = thunder_model(x)

torch.testing.assert_close(y, model(x))
```

## Examples

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

## Performance

Thunder delivers significant speedups compared to standard PyTorch eager execution.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community for support and collaboration:

*   💬 [Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)