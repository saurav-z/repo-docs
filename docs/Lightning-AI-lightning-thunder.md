# Lightning Thunder: Supercharge Your PyTorch Models ⚡

Lightning Thunder is a source-to-source compiler that empowers you to optimize and accelerate your PyTorch models with ease.  [Check out the original repo](https://github.com/Lightning-AI/lightning-thunder)!

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

**Key Features:**

*   **Accelerated Performance:** Achieve up to 40% faster PyTorch model execution.
*   **Simplified Optimization:** Easily integrate custom kernels, fusion, and quantization.
*   **Ready for Modern Hardware:** Optimized for NVIDIA Blackwell and other cutting-edge hardware.
*   **FP8/FP6/FP4 Precision:** Experiment with lower precision for significant speedups.
*   **Distributed Training & Inference:** Supports TP/PP/DP strategies.
*   **Broad Compatibility:** Works with LLMs, non-LLMs, and various model architectures.
*   **Custom Kernel Integration:**  Utilize Triton and CUDA kernels to further customize your models.
*   **Composable Transformations:**  Build complex optimizations by combining various techniques.
*   **CUDA Graphs Integration:**  Reduce CPU overhead with seamless CUDA Graphs support.
*   **User-Friendly Plugins:** Leverage out-of-the-box plugins for quick performance gains.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started with Lightning Thunder in a few simple steps:

1.  **Installation:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    For more installation options, including Blackwell support and advanced installations, refer to the [installation documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

2.  **Hello World Example:**

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

## Examples: Speed up your models

Lightning Thunder allows users to speed up LLMs and other models.

### Speed up LLM training

Install LitGPT (without updating other dependencies)

```
pip install --no-deps 'litgpt[all]'
```

and run

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

```
pip install -U transformers
```

and run

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

```
pip install -U transformers
```

and run

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

## Plugins

Thunder provides a plugin-based architecture for applying various optimizations. This enables fine-grained control over model acceleration.

### Available Plugins
*   **reduce-overhead**: CUDA graphs

## How Lightning Thunder Works

Thunder transforms your PyTorch models through three main stages:
1.  **Acquisition:** Thunder interprets your Python bytecode to create a straight-line Python program.
2.  **Transformation:**  The computation trace is transformed to enable distribution and precision changes.
3.  **Execution:**  Parts of the trace are routed for execution using techniques like fusion (NVFuser, torch.compile), specialized libraries, and custom kernels.

## Performance

Lightning Thunder delivers significant performance gains, particularly on modern hardware. See the provided pre-training task performance graphs on H100 and B200 hardware for detailed speedups.

## Community

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)