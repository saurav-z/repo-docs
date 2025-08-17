<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Thunder Logo" width="400px" style="max-width: 100%;">
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Thunder Logo" width="400px" style="max-width: 100%;">
</div>

# Lightning Thunder: Supercharge Your PyTorch Models with Ease

**Lightning Thunder** is a source-to-source compiler for PyTorch that simplifies model optimization, making your AI models faster, more efficient, and easier to understand.  [Get Started with Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder)

<br/>

**Key Features:**

*   🚀 **Significant Speedups:** Achieve up to 40% faster PyTorch model execution.
*   ⚙️ **Simplified Optimization:** Easily integrate custom kernels, fusions, quantization, and distributed strategies.
*   💡 **User-Friendly for Everyone:** Designed for both end-users seeking out-of-the-box performance improvements and performance experts aiming to understand and refine their models.
*   📦 **Pre-built plugins:** Integrate strategies for quantization, CUDA Graphs, FP8, FP6, and FP4 precision
*   🧠 **LLM & Beyond:**  Works with LLMs, non-LLMs and more
*   🛠️ **Extensive Support:** Works with NVIDIA Blackwell and CUDA Graphs

<div align="center">
  <pre>
  ✅ Up to 40% Faster PyTorch 🚀  ✅ Quantization                ✅ Kernel Fusion
  ✅ Training Recipes            ✅ FP4/FP6/FP8 Precision     ✅ Distributed TP/PP/DP
  ✅ Inference Recipes           ✅ NVIDIA Blackwell Ready    ✅ CUDA Graphs
  ✅ LLMs, non-LLMs & more      ✅ Custom Triton Kernels     ✅ Compose All the Above
  </pre>
</div>

<div align="center">
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
  [![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
  [![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
  [![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
  [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)
</div>

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick Start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

<br/>

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance Benchmarks" width="800px" style="max-width: 100%;">
</div>

## Quick Start

1.  **Install Thunder:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26

    pip install lightning-thunder
    ```

    See the [Lightning Thunder Installation Documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for additional install options.

2.  **Hello World Example:**

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

*   **Speed up LLM Training**

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

*   **Speed up HuggingFace BERT Inference**

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

*   **Speed up HuggingFace DeepSeek R1 distill inference**

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

*   **Speed up Vision Transformer Inference**

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

*   **Benchmarking HF Models**

    The script `examples/quickstart/hf_benchmarks.py` demonstrates how to benchmark a model for text generation, forward pass, forward pass with loss, and a full forward + backward computation.

## Plugins

Thunder's plugin system makes it easy to integrate various optimizations like:

*   Distributed strategies (DDP, FSDP, TP)
*   Numerical precision optimization (FP8, MXFP8)
*   Memory saving via quantization
*   Reduce latency with CUDAGraphs

    Use the `plugins=` argument of `thunder.compile` to enable features.  For example:
    ```python
    thunder_model = thunder.compile(model, plugins="reduce-overhead")
    ```

## How It Works

Thunder accelerates PyTorch models through these stages:

1.  **Acquisition:** Interprets Python bytecode to create a straight-line Python program.
2.  **Transformation:** Optimizes the computation trace (distribution, precision changes).
3.  **Execution Routing:** Routes parts of the trace to fusion engines (NVFuser, torch.compile), specialized libraries (cuDNN SDPA, TransformerEngine), custom Triton/CUDA kernels, or PyTorch eager operations.

<div align="center">
  <img src="docs/source/_static/images/how_it_works.png" alt="How Thunder Works" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers impressive speedups. The chart below shows performance gains on a pre-training task using LitGPT on H100 and B200 hardware.

## Community

Thunder is an open-source project.

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)