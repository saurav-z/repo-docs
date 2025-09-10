# Lightning Thunder: Supercharge Your PyTorch Models for Blazing-Fast Performance

**Lightning Thunder is a source-to-source compiler designed to dramatically accelerate your PyTorch models, making them faster, more efficient, and easier to optimize.**  [Check out the original repo](https://github.com/Lightning-AI/lightning-thunder)

<!-- Image Section with Light and Dark Mode Images -->
<div align="center">
  <img alt="Thunder - Light Mode" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder - Dark Mode" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

<!-- Summary -->
Lightning Thunder provides a flexible framework for optimizing PyTorch models with custom kernels, fusion, quantization, distributed strategies, and more.  Whether you're an end-user seeking immediate speedups or a performance expert looking for advanced customization, Lightning Thunder offers the tools you need.

**Key Features:**

*   ✅ **Accelerated Training and Inference:** Achieve up to 2x faster performance on your models.
*   ✅ **FP4/FP6/FP8 Precision:** Optimize numerical precision for improved throughput.
*   ✅ **Kernel Fusion:** Automatically fuse operations for enhanced efficiency.
*   ✅ **Distributed Training Strategies:** Leverage TP/PP/DP for scaling models.
*   ✅ **Quantization Support:** Reduce memory footprint and accelerate inference.
*   ✅ **CUDA Graphs:** Minimize CPU overhead for faster execution.
*   ✅ **Custom Kernel Integration:** Integrate Triton and CUDA kernels.
*   ✅ **Flexible Plugins:** Easily add and combine optimizations.

<div align='center'>
  <pre>
  ✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
  ✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
  ✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
  ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
  </pre>
</div>

**Quick Start:**

Get started in minutes with `pip`:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Advanced Install Options (Blackwell, additional executors, bleeding edge, dev):**

See the original README for detailed instructions: [https://github.com/Lightning-AI/lightning-thunder](https://github.com/Lightning-AI/lightning-thunder)

**Simple Example:**

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

**Examples**

*   [Speed up LLM training](#speed-up-llm-training)
*   [Speed up Hugging Face BERT inference](#speed-up-huggingface-bert-inference)
*   [Speed up Hugging Face DeepSeek R1 distill inference](#speed-up-huggingface-deepseek-r1-distill-inference)
*   [Speed up Vision Transformer inference](#speed-up-vision-transformer-inference)
*   [Benchmarking HF models](#benchmarking-hf-models)

## Speed up LLM training

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

## Speed up HuggingFace BERT inference

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

## Speed up HuggingFace DeepSeek R1 distill inference

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

## Speed up Vision Transformer inference

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

## Benchmarking HF models

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

**How Lightning Thunder Works:**

1.  **Acquisition:** Interprets Python bytecode to create a straight-line Python program.
2.  **Transformation:**  Transforms the computation trace for distribution and precision changes.
3.  **Execution Routing:**  Routes parts of the trace for optimized execution via:
    *   Fusion (NVFuser, torch.compile)
    *   Specialized Libraries (cuDNN SDPA, TransformerEngine)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
  <img alt="Thunder - How It Works" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

**Performance:**

See real-world speedups on pre-training tasks using LitGPT on H100 and B200 hardware in the performance graph in the original README (linked above).

**Community and Support:**

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)