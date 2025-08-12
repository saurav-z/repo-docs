<div align="center">

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

</div>

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>

</div>

**Lightning Thunder is a source-to-source compiler for PyTorch, offering unparalleled speed and efficiency for your AI models.**  Get ready to unlock the full potential of your models with features like custom kernels, model fusion, and more!  Explore the  [Lightning Thunder GitHub repository](https://github.com/Lightning-AI/lightning-thunder) for more details.

**Key Features:**

*   🚀 **Significant Speedups:** Achieve up to 40% faster PyTorch model execution.
*   💡 **Easy Optimization:** Simplify model optimization with composable transformations.
*   ⚙️ **Customizable:** Extend your models with custom kernels and flexible plugins.
*   🧪 **Precision Control:** Utilize FP4/FP6/FP8 precision for optimal performance.
*   🧠 **Distributed Training:** Leverage TP/PP/DP strategies for scalable training.
*   ⚡ **Model Fusion:**  Enhance performance with built-in kernel fusion.
*   ⚙️ **CUDA Graphs:** Reduce CPU overhead using CUDA Graphs.
*   🧠 **LLM and Beyond:**  Supports large language models and various other model types.
*   🐍 **Python-based:** Built and optimized for PyTorch models.
*   🔄 **Integrations:** Compatible with NVIDIA Blackwell and other advanced hardware.

---

<div align='center'>

✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above

</div>

<div align='center'>

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
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
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

&#160;

<!--
<div align="center">
<a target="_blank" href="https://lightning.ai/docs/thunder/home/get-started">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>
</div>
-->

&#160;

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

**Get started with Lightning Thunder in minutes:**

1.  **Install:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

2.  **Integrate into your code:**

    ```python
    import torch.nn as nn
    import thunder
    import torch

    # Define your model
    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

    # Compile your model with Thunder
    thunder_model = thunder.compile(model)

    # Run the compiled model
    x = torch.randn(64, 2048)
    y = thunder_model(x)

    torch.testing.assert_close(y, model(x))
    ```

    See the installation and getting started guides on [Lightning's Documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for more advanced options.

<details>
  <summary>Advanced Install Options</summary>

  ### Blackwell Support

  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
  pip install lightning-thunder
  ```

  ### Install additional executors

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

**Explore practical applications of Lightning Thunder:**

*   **Speed up LLM Training:** Fine-tune your large language models more efficiently.
*   **Accelerate Hugging Face BERT Inference:** Achieve faster inference speeds with pre-trained models.
*   **Optimize Hugging Face DeepSeek R1 Distill Inference:** Leverage Thunder for fast inference.
*   **Boost Vision Transformer Inference:** Accelerate your computer vision models.

```bash
# Example: Speed up LLM training
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

```bash
# Example: Speed up HuggingFace BERT inference
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

```bash
# Example: Speed up HuggingFace DeepSeek R1 distill inference
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

To see the speedups, just run:

```bash
python examples/quickstart/hf_llm.py
```

**Example Results:**

```bash
Eager: 2273.22ms
Thunder: 1254.39ms
```

81% faster 🏎️! Quite the speedup ⚡️

## Plugins

**Enhance your models with powerful plugins:**

*   **Reduce Overhead:**  Reduce CPU overhead via CUDAGraphs
*   **Distributed Training:** Use plugins for DDP, FSDP, and TP.
*   **Precision Tuning:**  Apply FP8 and MXFP8 for numerical optimizations.
*   **Memory Saving:** Implement quantization strategies.
*   **Debugging and Profiling:** Easily identify bottlenecks and areas for improvement.

To use plugins, simply pass the desired plugin names to the `plugins` argument in the `thunder.compile` function.  For example:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How It Works

**Lightning Thunder's core architecture:**

1.  **Acquisition:** Interprets Python bytecode to generate a straight-line Python program.
2.  **Transformation:** Transforms the computation trace for distribution and precision changes.
3.  **Execution:** Routes parts of the trace for execution, including fusion, specialized libraries, custom kernels, and PyTorch eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

```python
# Example of an acquired trace for a simple MLP
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

**Lightning Thunder delivers significant performance gains:**

*   **Pre-training Task on H100 & B200:** Demonstrates substantial speedups over PyTorch eager execution.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is developed collaboratively, with significant contributions from the community and NVIDIA.

💬 [Join the Discussion on Discord](https://discord.com/invite/XncpTy7DSt)

📜 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)

```