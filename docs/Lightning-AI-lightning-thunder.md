<!-- Improved & Summarized README - SEO Optimized -->
<!-- Start of Improved README -->

<div align="center">
  <h1>Lightning Thunder: Supercharge Your PyTorch Models ⚡</h1>
  <p><b>Accelerate your PyTorch models by up to 40% with Lightning Thunder, a source-to-source compiler.</b></p>
</div>

<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" alt="Lightning Thunder" width="400px" style="max-width: 100%;">
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" alt="Lightning Thunder" width="400px" style="max-width: 100%;">
</div>

<br>

---

**Lightning Thunder** is your key to unlocking unparalleled performance in PyTorch.  This powerful source-to-source compiler empowers both end-users and performance experts with easy optimization.  It's fast, understandable, and extensible, and will help you to push the boundaries of what's possible with your AI models.  Check out the original repo:  [Lightning-AI/lightning-thunder](https://github.com/Lightning-AI/lightning-thunder).

### Key Features

*   🚀 **Significant Speedups:** Achieve up to 40% faster model execution.
*   ⚙️ **Model Optimization:** Easily incorporate custom kernels, fusions, quantization (FP4/FP6/FP8), and distributed strategies.
*   🛠️ **Pre-built Plugins:** Benefit from ready-to-use plugins for NVIDIA Blackwell, CUDA Graphs, and more.
*   🧠 **Extensible & Customizable:** The framework provides the tools to understand, modify, and optimize AI models.
*   💡 **Comprehensive Support:** Features FP4/FP6/FP8 precision, distributed TP/PP/DP, and custom Triton kernels.
*   🔄 **Compose Optimizations:** Combine multiple optimizations for maximum impact.
*   💻 **Compatible with LLMs and More:** Thunder supports a wide variety of models, including LLMs and other architectures.

<div align="center">
  <pre>
  ✅ Up to 40% Faster PyTorch Performance    ✅ Quantization
  ✅ Training and Inference Recipes           ✅ FP4/FP6/FP8 Support
  ✅ NVIDIA Blackwell Ready                    ✅ CUDA Graphs
  ✅ LLMs, Non-LLMs & More                   ✅ Custom Triton Kernels
  </pre>
</div>

<div align="center">
  <a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml"><img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push" alt="CI Testing"></a>
  <a href="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml"><img src="https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push" alt="General Checks"></a>
  <a href="https://lightning-thunder.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/lightning-thunder/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main"><img src="https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg" alt="pre-commit.ci status"></a>
</div>

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick Start</a>
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a>
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a>
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

---

<div align="center">
  <img src="docs/source/_static/images/pretrain_perf.png" alt="Performance Benchmark" width="800px" style="max-width: 100%;">
</div>

### Quick Start - Get Started Today!

**Installation**:  Start by installing the necessary dependencies, then Thunder itself:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

<details>
  <summary>Advanced Installation Options</summary>
  <!-- Installation details here -->
  <br>
  **Blackwell Support**:

  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
  pip install lightning-thunder
  ```

  **Install Additional Executors**:

  ```bash
  # cuDNN SDPA
  pip install nvidia-cudnn-frontend

  # Float8 support (this will compile from source, be patient)
  pip install "transformer_engine[pytorch]"
  ```

  **Install Thunder Bleeding Edge**:

  ```bash
  pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
  ```

  **Install Thunder for Development**:

  ```bash
  git clone https://github.com/Lightning-AI/lightning-thunder.git
  cd lightning-thunder
  pip install -e .
  ```
</details>

**Hello World Example:**  Compile your PyTorch model with Thunder and run!

```python
import torch.nn as nn
import thunder
import torch

# Define your model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile with Thunder
thunder_model = thunder.compile(model)

# Run inference
x = torch.randn(64, 2048)
y = thunder_model(x)

# Verify results
torch.testing.assert_close(y, model(x))
```

### Examples - Accelerate your Model

**1. Speed Up LLM Training:**

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

**2. Speed Up HuggingFace BERT Inference:**

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

**3. Speed Up HuggingFace DeepSeek R1 Distill Inference:**

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

Run `python examples/quickstart/hf_llm.py` to benchmark. Here's a result from a L4 machine:

```bash
Eager: 2273.22ms
Thunder: 1254.39ms
```
An 81% speedup! 🏎️

**4. Speed Up Vision Transformer Inference:**

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

**5. Benchmarking HF models**

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

### Plugins - Power Up Your Models

Thunder's plugins provide modular optimizations for performance.

*   **Distributed Strategies:**  Scale up with DDP, FSDP, TP (Tensor Parallelism).
*   **Numerical Precision:** Optimize with FP8 and MXFP8.
*   **Memory Efficiency:** Utilize Quantization techniques.
*   **Reduce Latency:** Implement CUDA Graphs.
*   **Debugging and Profiling:** Simplify debugging and profile your models

**Example: Use CUDA Graphs**

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

### How It Works

Thunder operates in three key stages:

1.  **Acquisition:** Interprets Python bytecode to create a straight-line program.
2.  **Transformation:** Optimizes the computation trace for distribution and precision.
3.  **Execution Routing:**  Executes the trace with:
    *   Fusion (NVFuser, torch.compile)
    *   Specialized libraries (cuDNN SDPA, TransformerEngine)
    *   Custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
  <img src="docs/source/_static/images/how_it_works.png" alt="How Thunder Works" width="800px" style="max-width: 100%;">
</div>

This is the acquired trace for a simple MLP:
```python
import thunder
import torch.nn as nn

model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 256))

thunder_model = thunder.compile(model)
y = thunder_model(torch.randn(4, 1024))

print(thunder.last_traces(thunder_model)[-1])
```

This is the acquired trace, ready to be transformed and executed:

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
Thunder's intermediate representation is just (a subset of) Python!

### Performance - See the Speed!

Experience significant speedups on your pre-training tasks using LitGPT on both H100 and B200 hardware.  Refer to the performance chart above for detailed results.

### Community - Join Us!

Thunder is an open-source project built with community collaboration.

*   💬 [Join our Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)

<!-- End of Improved README -->