# Lightning Thunder: Supercharge Your PyTorch Models for Peak Performance

**Accelerate your PyTorch models with Lightning Thunder, a source-to-source compiler that unlocks significant speedups through optimizations like kernel fusion, quantization, and distributed training.**

[Lightning Thunder on GitHub](https://github.com/Lightning-AI/lightning-thunder)

---

## Key Features

*   ⚡ **Fast Execution:** Achieve up to 81% faster training and inference speeds.
*   🔥 **Model Optimization:** Leverage custom kernels, fusions, quantization, and various distributed strategies.
*   🛠️ **Extensible Architecture:** Easily add custom transformations and plugins to tailor your optimization strategy.
*   🎯 **Precision Control:** Utilize FP4/FP6/FP8 precision for memory efficiency and speed.
*   🚀 **Blackwell Ready:** Optimized for the latest NVIDIA Blackwell hardware.
*   📚 **Versatile Compatibility:** Supports LLMs, non-LLMs, and a wide range of PyTorch models.

---

## Quick Start

Get started with Lightning Thunder in just a few steps!

### Installation

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Note:** For specific hardware and advanced options, refer to the [installation documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

### Basic Usage

Optimize your PyTorch models with a single line of code:

```python
import thunder
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
thunder_model = thunder.compile(model)

x = torch.randn(64, 2048)
y = thunder_model(x)

torch.testing.assert_close(y, model(x))
```

---

## Examples

### Speeding up LLM Training

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

### Speeding up Hugging Face BERT inference

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

### Benchmarking HF models

```bash
python examples/quickstart/hf_benchmarks.py
```

---

## Plugins

Lightning Thunder offers a plugin system to customize and extend your optimization strategies:

*   **Distributed Strategies:** DDP, FSDP, TP.
*   **Numerical Precision:** FP8, MXFP8.
*   **Quantization:** Save memory with quantization.
*   **CUDA Graphs:** Reduce CPU overhead.
*   **Profiling and Debugging:** Analyze and optimize your models.

Example: Enable CUDA Graphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

---

## Performance

Lightning Thunder delivers significant performance improvements:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

---

## Community

*   💬 [Join our Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)
*   [Documentation](https://lightning.ai/docs/thunder/latest/)