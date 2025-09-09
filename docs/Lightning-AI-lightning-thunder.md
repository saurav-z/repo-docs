<!--
  SPDX-FileCopyrightText: 2024 Lightning AI, Inc.
-->

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder is a source-to-source compiler that accelerates PyTorch models, offering easy optimization for both beginners and performance experts.** [Check out the GitHub repository for more details.](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features:

*   🚀 **Significant Speedups:** Achieve up to 40% faster PyTorch model execution.
*   🛠️ **Flexible Optimization:**  Integrate custom kernels, fusions, quantization, and distributed strategies.
*   🎯 **Ready-to-Use Plugins:**  Leverage pre-built plugins for immediate performance gains.
*   ⚖️ **Precision Tuning:** Optimize numerical precision with FP4/FP6/FP8 support.
*   🌐 **Distributed Training:**  Implement efficient distributed training with TP/PP/DP.
*   💡 **CUDA Graph Integration:**  Reduce CPU overheads and boost performance with CUDA Graphs.
*   🔥 **Hardware Ready:**  Optimized for NVIDIA Blackwell and other cutting-edge hardware.
*   🧠 **Understandable & Extensible:** Easily modify and optimize AI models through composable transformations.
*   ✅ **LLM & Non-LLM Support:** Works with a wide range of models, including LLMs.

<div align="center">
<pre>
✅ Run PyTorch models faster     ✅ FP4/FP6/FP8 precision    ✅ CUDA Graphs
✅ Training & Inference recipes  ✅ Quantization             ✅ Triton kernels
✅ Distributed TP/PP/DP        ✅ Kernel fusion            ✅ Compose all the above
</pre>
</div>

## Get Started Quickly

1.  **Install Thunder:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    See the [installation docs](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for advanced options like Blackwell support.

2.  **Optimize Your Model:**

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

Explore example code demonstrating how to supercharge your models.

*   **Speed up LLM training**
    ```bash
    pip install --no-deps 'litgpt[all]'
    ```
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
*   **Speed up HuggingFace BERT inference**
    ```bash
    pip install -U transformers
    ```
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
    ```bash
    pip install -U transformers
    ```
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
*   **Speed up Vision Transformer inference**
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

## Benchmarks

Run the following command for benchmarks on HF models.
```bash
python examples/quickstart/hf_benchmarks.py
```

## Plugins for Advanced Control

Enhance your model optimization process using Lightning Thunder's plugin system.  Plugins provide modularity and flexibility, allowing you to easily experiment with different optimizations.

*   **reduce-overhead** For reduce CPU overheads via CUDAGraphs you can add "reduce-overhead"
    to the `plugins=` argument of `thunder.compile`:
```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How Lightning Thunder Works

Lightning Thunder streamlines the optimization process through these key stages:

1.  **Acquisition:**  Interprets Python bytecode to create a straight-line program representation.
2.  **Transformation:** Transforms the computation trace for optimizations like distribution and precision changes.
3.  **Execution:** Routes trace components for fusion, specialized libraries (e.g. cuDNN SDPA, TransformerEngine), custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance Highlights

See substantial performance gains with Lightning Thunder.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community & Resources

*   💬 [Join the Discord](https://discord.com/invite/XncpTy7DSt)
*   📜 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)