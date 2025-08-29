<!--
  SPDX-License-Identifier: Apache-2.0
-->

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

[Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) empowers you to optimize your PyTorch models with ease, unlocking significant performance gains.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   🚀 **Up to 8x Faster Training & Inference:** Experience dramatic speed improvements with Thunder's optimized execution.
*   ⚙️ **Automated Optimizations:** Enhance your models with just a few lines of code, including kernel fusion, quantization, and distributed strategies.
*   💡 **Easy to Use:** Thunder provides ready-to-use plugins for performance gains out of the box.
*   🧩 **Composable Transformations:**  Fine-tune and optimize your models through composable transformations.
*   🔬 **FP4/FP6/FP8 Precision:**  Leverage lower precision formats for memory savings and faster computations.
*   🌐 **Distributed Training:** Supports data parallelism (DP), tensor parallelism (TP) and pipeline parallelism (PP).
*   💻 **CUDA Graphs & Custom Kernels:** Enables low-latency execution and custom kernel integration for expert users.
*   🧠 **LLMs and Beyond:** Designed for both Large Language Models and other model architectures.

## Getting Started

### Installation

Install Thunder using pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Note:** Advanced installation options and support for Blackwell and other features can be found in the [original README](https://github.com/Lightning-AI/lightning-thunder).

### Quick Example: Compile a Simple Model

```python
import torch
import torch.nn as nn
import thunder

# Define a PyTorch model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Create input data
x = torch.randn(64, 2048)

# Run the compiled model
y = thunder_model(x)

# Verify output
torch.testing.assert_close(y, model(x))
```

## Examples

Thunder seamlessly integrates with popular models and frameworks, offering performance improvements in various scenarios:

*   **Speed up LLM Training:**  Integrate with LitGPT for faster LLM training.
*   **Accelerate Hugging Face Inference:** Boost inference speeds for Hugging Face models.
*   **Optimize Vision Transformers:**  Enhance Vision Transformer performance.
*   **Model Benchmarking:** Use Thunder's benchmarking tools to compare performance.

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

## Performance

Thunder delivers substantial performance gains. See below how a Llama-3.2-1B is performing on an A100:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Plugins

Thunder's plugin architecture enables flexible optimization:

*   **Reduce CPU overhead:** Utilize CUDA Graphs to improve performance.
*   **Distributed Training:** Scale up models with various distributed strategies.
*   **Precision:** Optimize for FP8 precision with MXFP8 and TransformerEngine
*   **Quantization:** Save memory with quantization.

## How it Works

Thunder's optimization process involves:

1.  **Trace Acquisition:** Interpreting Python bytecode.
2.  **Computation Transformation:** Applying transformations to create optimized code.
3.  **Execution Routing:**  Dispatching operations to optimized execution engines (fusion, specialized libraries, custom kernels).

## Community

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)