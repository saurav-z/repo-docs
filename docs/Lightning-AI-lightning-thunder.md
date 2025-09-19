# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Accelerate your PyTorch models with Lightning Thunder, a source-to-source compiler that unlocks significant performance gains and simplifies model optimization.** [Explore the original repo](https://github.com/Lightning-AI/lightning-thunder).

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

**Key Features:**

*   **Significant Speedups:** Achieve up to 81% faster performance out-of-the-box.
*   **Easy Optimization:** Utilize plugins for quantization, fusion, distributed training, and more.
*   **FP4/FP6/FP8 Precision:** Leverage lower precision to reduce memory footprint and increase speed.
*   **Support for Modern Hardware:** Optimized for NVIDIA Blackwell and CUDA Graphs.
*   **Custom Kernels:** Integrate custom Triton kernels for specialized operations.
*   **LLM and Non-LLM Compatibility:** Works with a wide range of models, including LLMs.
*   **Composable Transformations:** Customize and combine optimizations for maximum efficiency.

<div align='center'>
  ✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
  ✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
  ✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
  ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</div>

---

## Quick Start

Get started with Lightning Thunder in a few simple steps:

1.  **Install:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    **Advanced Install Options:** (See original README for detailed instructions on Blackwell Support and other options)

2.  **Hello World:**

    ```python
    import torch
    import torch.nn as nn
    import thunder

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
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

*   **Benchmarking HF models:**  The script `examples/quickstart/hf_benchmarks.py` provides benchmarks for a range of Hugging Face models.

## Plugins

Extend Thunder's functionality with plugins:

*   **Distributed Strategies:** DDP, FSDP, TP
*   **Numerical Precision:** FP8, MXFP8
*   **Memory Optimization:** Quantization
*   **Latency Reduction:** CUDAGraphs

Example: Enable CUDA Graphs for reduced CPU overhead:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder optimizes your models in three key stages:

1.  **Trace Acquisition:** Interprets Python bytecode and produces a straight-line Python program.
2.  **Transformation:** Modifies the computation trace for distribution and precision changes.
3.  **Execution Routing:** Directs parts of the trace to:
    *   Fusion (`NVFuser`, `torch.compile`)
    *   Specialized Libraries (`cuDNN SDPA`, `TransformerEngine`)
    *   Custom Triton/CUDA Kernels
    *   PyTorch eager operations

<div align="center">
    <img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers impressive performance gains.  See the pre-training task benchmarks on H100 and B200 hardware:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

---

## Community

Join the Thunder community:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)