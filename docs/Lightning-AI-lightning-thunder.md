# Lightning Thunder: Supercharge Your PyTorch Models

⚡ **Lightning Thunder accelerates PyTorch models with source-to-source compilation, unlocking significant performance gains.**  [Check out the original repository!](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

<div align="center">
  <pre>
✅ Run PyTorch up to 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes                 ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes                ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more          ✅ Custom Triton kernels       ✅ Compose all the above
  </pre>
</div>

## Key Features

*   **Simplified Optimization:** Easily integrate optimizations like kernel fusion, quantization, and distributed strategies.
*   **Out-of-the-Box Speedups:**  Benefit from pre-built plugins for immediate performance improvements on modern hardware.
*   **Extensible Architecture:**  A flexible framework for understanding, modifying, and optimizing AI models through composable transformations.
*   **Broad Compatibility:** Works with a wide range of models, including LLMs and vision models.
*   **Flexible Precision:** Supports FP4, FP6, and FP8 for optimized performance and memory usage.
*   **Advanced Techniques:**  Leverages CUDA Graphs, custom Triton kernels, and distributed training/inference strategies (TP/PP/DP) for peak efficiency.

## Quick Start

**Installation:**

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Advanced Installation Options & CUDA Support** (Refer to the original README for specific instructions on Blackwell and additional executor installations.)

**Hello World Example:**

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

**Speed up LLM Training with LitGPT:** (Requires LitGPT Installation. See original README for details.)

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

**Speed up HuggingFace BERT Inference:** (Requires Transformers Installation. See original README for details.)

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

**Speed up HuggingFace DeepSeek R1 Distill Inference** (Requires Transformers Installation. See original README for details.)

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
**Speed up Vision Transformer Inference**

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

## Plugins

Thunder plugins enable easy application of optimizations.  Example:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")  # For CUDA Graphs
```

## How it Works

Thunder operates through three key stages:

1.  **Acquisition:** Interprets Python bytecode to create a straight-line program.
2.  **Transformation:** Modifies the computation trace for distribution, precision changes, etc.
3.  **Execution Routing:**  Routes parts of the trace to optimized executors like NVFuser, custom kernels, and eager PyTorch operations.

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers significant speedups, especially on modern hardware. (See the image in the original README for pre-training performance.)

## Community

Join the Thunder community for support and collaboration!

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)