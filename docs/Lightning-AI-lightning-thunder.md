<!--
  _   _ _____   _____  _  _____  _    _  _____
 | | | |_   _| / ____|| ||_   _|| |  | ||  ___|
 | | | | | |  | (___  | |  | |  | |  | || |__
 | | | | | |   \___ \ | |  | |  | |  | ||  __|
 | |_| |_| |_  ____) || |_ | |_ | |_/ /| |___
  \___/|_____||_____/  \__| \__| \___/ \____/
-->

<div align="center">

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

</div>

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>

**Lightning Thunder is a source-to-source compiler for PyTorch, enabling significant performance boosts and optimized model execution.**

</div>

---

Lightning Thunder empowers you to easily optimize your PyTorch models with cutting-edge features and techniques. Whether you're a beginner or a seasoned expert, Thunder provides the tools to achieve significant performance gains.

**Key Features:**

*   🚀 **Accelerated Performance:** Experience up to 40% faster model execution with minimal code changes.
*   💡 **Simplified Optimization:** Benefit from ready-to-use plugins for model speedups and optimal hardware utilization.
*   🔬 **Customization & Extensibility:** Build and modify AI models with composable transformations.
*   ⚙️ **Advanced Techniques:** Leverage quantization, FP4/FP6/FP8 precision, distributed training strategies (TP/PP/DP), and more.
*   🛠️ **Cutting-Edge Support:** Ready for NVIDIA Blackwell and utilizes CUDA Graphs for enhanced efficiency.
*   🔥 **Triton Integration:** Integrate custom Triton kernels for unparalleled performance.
*   🔄 **Composable Transformations:** Easily combine features for optimal results.

<div align='center'>

<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>

</div>

<div align='center'>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
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
    <a target="_blank" href="https://github.com/Lightning-AI/lightning-thunder" style="margin: 0 10px;">View on GitHub</a>
  </div>
</div>

---

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in a few steps:

1.  **Installation:** Install Thunder and its dependencies using pip ([more options](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html)):

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
2.  **Integrate into your Code:** Apply `thunder.compile` to your model for immediate speedups:

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

<details>
  <summary>Advanced Installation Options</summary>

### Blackwell Support

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

## Examples

Explore these examples to see Thunder in action:

### Speed Up LLM Training

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

### Speed Up HuggingFace BERT Inference

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

### Speed Up HuggingFace DeepSeek R1 Distill Inference

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

Here's an example on a L4 machine from [Lightning Studio](https://lightning.ai):

```bash
Eager: 2273.22ms
Thunder: 1254.39ms
```

Which is 81% faster! 🏎️

### Speed Up Vision Transformer Inference

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

Thunder's plugins provide easy-to-use optimizations, such as parallelism and quantization.  Easily incorporate various optimizations.

*   Scale up with distributed strategies with DDP, FSDP, TP ()
*   Optimize numerical precision with FP8, MXFP8
*   Save memory with quantization
*   Reduce latency with CUDAGraphs
*   Debugging and profiling

For instance, reduce CPU overheads using CUDAGraphs. Just add "reduce-overhead" to `plugins=` argument of `thunder.compile`:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder transforms your PyTorch models in three stages:

1.  **Acquisition:** Interprets Python bytecode to produce a straight-line Python program.
2.  **Transformation:** Optimizes the computation trace by distributing and/or changing precision.
3.  **Execution:** Routes parts of the trace for execution via fusion (NVFuser, torch.compile), specialized libraries (cuDNN SDPA, TransformerEngine), custom Triton and CUDA kernels, or standard PyTorch operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

Example of a simple MLP trace:

```python
import thunder
import torch.nn as nn

model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 256))

thunder_model = thunder.compile(model)
y = thunder_model(torch.randn(4, 1024))

print(thunder.last_traces(thunder_model)[-1])
```

The acquired trace:

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

Thunder delivers significant performance gains. See below how Thunder performs compared to PyTorch eager on various hardware.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is open source, developed with community contributions and support from NVIDIA.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)