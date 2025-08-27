# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder is a source-to-source compiler that transforms your PyTorch models, providing significant speedups, quantization, and more for both beginners and performance experts.**  [Explore the Lightning Thunder GitHub Repo](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>
</div>

## Key Features

*   **Accelerated Performance:** Experience up to 40% faster PyTorch model execution.
*   **Optimized Precision:** Utilize FP4/FP6/FP8 precision for efficient training and inference.
*   **Advanced Optimization:** Benefit from features like Kernel fusion, CUDA Graphs, and distributed training strategies (TP/PP/DP).
*   **Ready for Next-Gen Hardware:** Optimized for NVIDIA Blackwell and other advanced hardware.
*   **Customizable:** Leverage custom Triton kernels for tailored performance enhancements.
*   **Easy to Use:**  Ready-to-use plugins for out-of-the-box speedups.
*   **Composable Transformations:** A framework for understanding, modifying, and optimizing AI models.

<div align='center'>
  <pre>
  ✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
  ✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
  ✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
  ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
  </pre>
</div>

## Quick Start

Get started with Lightning Thunder quickly using pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

### Advanced Installation

Expand for advanced installation options.

<details>
  <summary>Advanced install options</summary>

#### Blackwell support

For Blackwell you'll need CUDA 12.8

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com

pip install lightning-thunder
```

#### Install additional executors

These are optional, feel free to mix and match

```bash
# cuDNN SDPA
pip install nvidia-cudnn-frontend

# Float8 support (this will compile from source, be patient)
pip install "transformer_engine[pytorch]"
```

#### Install Thunder bleeding edge

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
```

#### Install Thunder for development

```bash
git clone https://github.com/Lightning-AI/lightning-thunder.git
cd lightning-thunder
pip install -e .
```

</details>

### Hello World Example

Optimize your PyTorch models with Thunder in just a few lines:

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

Learn how to accelerate different model types.

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

Plugins extend Thunder's capabilities, providing easy access to advanced optimizations.  Enhance your models with:

*   Distributed Training (DDP, FSDP, TP)
*   Numerical Precision Optimization (FP8, MXFP8)
*   Memory Saving with Quantization
*   Reduced Latency via CUDAGraphs
*   Debugging and Profiling Tools

Add "reduce-overhead" via the `plugins=` argument of `thunder.compile`

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder employs a three-stage process to optimize your models:

1.  **Acquisition:** Interprets Python bytecode and generates a straight-line Python program.
2.  **Transformation:** Modifies the computation trace for distribution and precision adjustments.
3.  **Execution:** Routes the trace for execution via fusion, specialized libraries (e.g., cuDNN SDPA, TransformerEngine), custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Achieve significant speedups with Lightning Thunder.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is an open-source project driven by community collaboration, with significant contributions from NVIDIA.

*   💬 [Join the Discussion on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)