<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder">
    <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Logo" width="400" height="auto">
    <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Logo" width="400" height="auto">
  </a>
  <br />
  <a href="https://github.com/Lightning-AI/lightning-thunder">
    <img src="https://img.shields.io/github/stars/Lightning-AI/lightning-thunder?style=social" alt="Stars">
  </a>
  <p align="center">
    Supercharge your PyTorch models with Lightning Thunder, a source-to-source compiler that unlocks incredible performance gains.
    <br />
    <a href="https://github.com/Lightning-AI/lightning-thunder"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://discord.com/invite/XncpTy7DSt">Join our Discord</a>
    ·
    <a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE">License</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#plugins">Plugins</a></li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#performance">Performance</a></li>
    <li><a href="#community">Community</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Lightning Thunder is a source-to-source compiler designed to dramatically accelerate your PyTorch models. By leveraging cutting-edge techniques like custom kernels, fusions, quantization, and optimized distributed strategies, Thunder empowers both end-users and performance experts to achieve unparalleled efficiency.  The project is open-source and developed in collaboration with the community.

### Key Features

*   ⚡ **Significant Speedups:** Experience up to 81% faster performance in LLM inference and notable gains in various tasks.
*   ✅ **Optimized for Modern Hardware:** Ready for NVIDIA Blackwell and other latest-generation hardware, maximizing utilization.
*   🛠️ **Composable Transformations:** Build and customize AI models with composable transformations for performance experts.
*   ⚙️ **Comprehensive Optimization:** Includes quantization (FP4/FP6/FP8), kernel fusion, CUDA Graphs, distributed training (TP/PP/DP), and more.
*   💡 **User-Friendly Plugins:** Out-of-the-box plugins for ease of use.
*   🌐 **Flexible and Extensible:** Supports LLMs, non-LLMs, and custom Triton kernels.

## Quick Start

Get started with Thunder in a few steps:

1.  **Install Dependencies:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

2.  **Install Additional Executors (Optional):**

    ```bash
    # cuDNN SDPA
    pip install nvidia-cudnn-frontend

    # Float8 support (this will compile from source, be patient)
    pip install "transformer_engine[pytorch]"
    ```

3.  **Simple Example:**

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

    See [installation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for advanced installation options including Blackwell support, bleeding edge, and development installs.

## Examples

Thunder provides easy-to-use examples to get you started:

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

*   **Speed up Hugging Face BERT inference**

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

*   **Benchmarking HF models**

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

Thunder's plugin system enables flexible model optimization. Utilize pre-built plugins for distributed strategies (DDP, FSDP, TP), numerical precision (FP8, MXFP8), quantization, and reduced latency with CUDA Graphs. You can even create your own custom plugins. For example, use the `reduce-overhead` plugin for CUDA graphs.

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

Thunder optimizes models in three key stages:

1.  **Acquisition:** Thunder interprets Python bytecode, producing a straight-line Python program.
2.  **Transformation:** The computation trace is transformed for distribution and precision adjustments.
3.  **Execution:** Parts of the trace are routed for execution, leveraging:

    *   Fusion (NVFuser, torch.compile)
    *   Specialized libraries (e.g., cuDNN SDPA, TransformerEngine)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

    Here's a look at the trace for a simple MLP:

    ```python
    import thunder
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 256))

    thunder_model = thunder.compile(model)
    y = thunder_model(torch.randn(4, 1024))

    print(thunder.last_traces(thunder_model)[-1])
    ```

    Acquired Trace:

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

Thunder delivers significant performance gains.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join our community and contribute to the development of Thunder!

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)

<!-- FOOTER -->
<p align="right">(<a href="#top">back to top</a>)</p>