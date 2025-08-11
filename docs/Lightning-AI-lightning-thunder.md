<!-- Improved README for Lightning Thunder -->

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
  <br/>
  <br/>
  <!-- Optimized SEO Title and Description -->
  <h1>Lightning Thunder: Supercharge Your PyTorch Models for Blazing-Fast Performance</h1>
  <p><b>Lightning Thunder</b> is a source-to-source compiler for PyTorch that makes your models run faster, easier, and more efficiently, offering a suite of optimizations for diverse hardware.</p>
  <p>
    <a href="https://github.com/Lightning-AI/lightning-thunder">View on GitHub</a>
  </p>
</div>

---

## Key Features of Lightning Thunder

*   ⚡ **Accelerated Performance:** Achieve significant speedups (up to 81% faster!) by optimizing your PyTorch models.
*   🛠️ **Extensible Optimization:** Easily integrate custom kernels, fusions, quantization, and distributed strategies.
*   🧠 **User-Friendly for All:** Streamlined for both end-users seeking out-of-the-box speedups and performance experts looking for composable transformations.
*   🚀 **Pre-Built Plugins:** Ready-to-use plugins for various optimizations including FP4/FP6/FP8 precision, distributed training, and CUDA Graphs.
*   ⚙️ **Versatile Support:** Works with LLMs, non-LLMs, and is optimized for NVIDIA Blackwell architecture.
*   🔬 **Composable Transformations:** Understand, modify, and optimize AI models with composable transformations.

---

## Why Choose Lightning Thunder?

Lightning Thunder transforms your PyTorch models with a focus on performance, ease of use, and flexibility. Whether you're looking to quickly accelerate your existing models or dive deep into custom optimizations, Thunder provides the tools and capabilities you need. It's perfect for:

*   **Model Training:** Training recipes and significant performance gains to cut down on training time.
*   **Model Inference:** Inference recipes for rapid model deployment.
*   **Hardware Optimization:** Ready for NVIDIA Blackwell hardware and supports a broad range of hardware.

---

## Quick Start

Get up and running with Lightning Thunder in a few simple steps. For full installation options, see the [installation documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

pip install lightning-thunder
```

### Hello World Example

Here's how to quickly optimize a model:

```python
import torch
import torch.nn as nn
import thunder

# Define a simple model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Create input data
x = torch.randn(64, 2048)

# Run the model and compare
y = thunder_model(x)

#Verify output
torch.testing.assert_close(y, model(x))
```

---

## Examples

Lightning Thunder can be used to speed up different models with different optimizations.

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

## Performance

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)
*   📚 [Read the Docs](https://lightning-thunder.readthedocs.io/en/latest/)