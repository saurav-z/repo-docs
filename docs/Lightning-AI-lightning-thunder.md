# Lightning Thunder: Supercharge Your PyTorch Models with Ease

**Lightning Thunder** is a source-to-source compiler for PyTorch that empowers you to optimize your models with custom kernels, fusions, quantization, and distributed strategies for unparalleled performance. [Explore the Lightning Thunder Repository](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
</div>

## Key Features

*   ⚡ **Blazing Fast Performance**: Achieve significant speedups (up to 40% faster) in PyTorch model execution.
*   ⚙️ **Optimized for Modern Hardware**: Unleash the full potential of your NVIDIA hardware, including support for Blackwell.
*   🚀 **Easy to Use**: Simple API for integrating Thunder into your existing PyTorch workflows.
*   🧮 **Advanced Optimization Techniques**: Leverage quantization (FP4/FP6/FP8), kernel fusion, CUDA Graphs, and more.
*   🧠 **Customization and Extensibility**: Build custom Triton kernels and easily integrate new optimizations.
*   🌐 **Distributed Training and Inference**: Scale your models with support for TP/PP/DP distributed strategies.
*   🧩 **Pre-built Plugins**: Use out-of-the-box plugins for common optimizations.
*   💡 **Supports Diverse Models**: Works with LLMs, non-LLMs, and a wide range of PyTorch models.

<div align='center'>
<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>
</div>

## Quick Start

Get started with Lightning Thunder in a few simple steps:

1.  **Install Dependencies**:

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    **Note:**  For CUDA 12.8 and Blackwell support, use:

    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
    pip install lightning-thunder
    ```

2.  **Import and Compile Your Model**:

    ```python
    import thunder
    import torch.nn as nn
    import torch

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

## Examples

*   **Speed up LLM Training**:

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

*   **Speed up Hugging Face BERT inference**

    ```python
    import thunder
    import torch
    import transformers
    model_name = "bert-large-uncased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    with torch.device("cuda"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.eval()
        inp = tokenizer(["Hello world!"], return_tensors="pt")
    thunder_model = thunder.compile(model)
    out = thunder_model(**inp)
    print(out)
    ```

*   **Speed up Hugging Face DeepSeek R1 distill inference**

    ```python
    import torch
    import transformers
    import thunder
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    with torch.device("cuda"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.eval()
        inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt")
    thunder_model = thunder.compile(model)
    out = thunder_model.generate(
        **inp, do_sample=False, cache_implementation="static", max_new_tokens=100)
    print(out)
    ```

    **Get Speedups:**

    ```bash
    python examples/quickstart/hf_llm.py
    ```

    **Results Example:**

    ```bash
    Eager: 2273.22ms
    Thunder: 1254.39ms
    ```

*   **Speed up Vision Transformer inference**:

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

*   **Benchmarking HF models**
    See  `examples/quickstart/hf_benchmarks.py` for how to benchmark a model for text generation, forward pass, forward pass with loss, and a full forward + backward computation.

## Plugins

Customize your optimizations with Thunder plugins:

*   **Reduce CPU overheads with CUDAGraphs:**

    ```python
    thunder_model = thunder.compile(model, plugins="reduce-overhead")
    ```

## How it Works

Lightning Thunder works in three stages:

1.  **Acquisition**: Interprets Python bytecode to produce a straight-line Python program.
2.  **Transformation**: Transforms the computation trace for distribution and precision changes.
3.  **Execution**: Routes parts of the trace for execution using fusion, specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)