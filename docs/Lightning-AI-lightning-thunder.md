<!-- Add a meta description tag for SEO -->
<meta name="description" content="Supercharge your PyTorch models with Lightning Thunder: a source-to-source compiler for faster training and inference, offering custom kernels, quantization, and more.">

# Lightning Thunder: Supercharge Your PyTorch Models

Lightning Thunder is a cutting-edge source-to-source compiler for PyTorch, designed to dramatically accelerate model training and inference.  **[Explore Lightning Thunder on GitHub](https://github.com/Lightning-AI/lightning-thunder)**.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   ✅ **Significant Speedups:** Achieve up to 40% faster PyTorch model execution.
*   ✅ **Model Optimization:** Leverage custom kernels, kernel fusion, and quantization for optimal performance.
*   ✅ **Flexible Precision:** Supports FP4/FP6/FP8 precision for efficient training.
*   ✅ **Distributed Training:** Integrated support for various distributed training strategies like TP/PP/DP.
*   ✅ **Ready for Latest Hardware:** Optimized for NVIDIA Blackwell and other latest generation hardware.
*   ✅ **Composable Transformations:** Provides a framework for understanding, modifying, and optimizing AI models.
*   ✅ **Easy to Use:** Provides out-of-the-box plugins for immediate speed improvements.
*   ✅ **Custom Kernels:** Enables the integration of custom Triton kernels for specialized operations.

<div align='center'>
  <pre>
  ✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
  ✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
  ✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
  ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
  </pre>
</div>

## Get Started

### Installation

Install Lightning Thunder using pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

For more advanced installation options (Blackwell support, additional executors, bleeding-edge versions, and development setup), please refer to the [installation instructions](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) in the documentation.

### Quick Example:

```python
import torch
import torch.nn as nn
import thunder

# Define a PyTorch model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Create an input tensor
x = torch.randn(64, 2048)

# Run the compiled model
y = thunder_model(x)

# Verify results (optional)
assert torch.testing.assert_close(y, model(x))
```

## Examples

### Speed Up LLM Training

Install LitGPT:

```bash
pip install --no-deps 'litgpt[all]'
```

and run:

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

Install Hugging Face Transformers (recommended version is `4.50.2` and above)

```bash
pip install -U transformers
```

and run:

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

Install Hugging Face Transformers (recommended version is `4.50.2` and above)

```bash
pip install -U transformers
```

and run:

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

For speedup comparison run:
```bash
python examples/quickstart/hf_llm.py
```
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

Thunder's plugin system enables easy integration of various optimizations, such as:

*   **Distributed Training:** DDP, FSDP, TP (Tensor Parallelism)
*   **Numerical Precision:** FP8, MXFP8
*   **Quantization:**  Reduce memory footprint and improve speed.
*   **CUDAGraphs:** Reduce CPU overhead and latency.
*   **Profiling and Debugging:** Tools to analyze and optimize your models.

To enable a plugin (e.g., reduce CPU overhead with CUDAGraphs):

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder employs a three-stage process:

1.  **Acquisition:**  Interprets Python bytecode to create a streamlined program.
2.  **Transformation:** Optimizes the computation trace, enabling distributed training and precision changes.
3.  **Execution:** Routes the optimized trace using various techniques, including:
    *   Fusion (NVFuser, torch.compile)
    *   Specialized libraries (e.g., cuDNN SDPA, TransformerEngine)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

[See Performance image in original README]

## Community

Lightning Thunder is an open-source project developed with significant community contributions.

*   💬 [Join the Discord Server](https://discord.com/invite/XncpTy7DSt)
*   📜 [License: Apache 2.0](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)