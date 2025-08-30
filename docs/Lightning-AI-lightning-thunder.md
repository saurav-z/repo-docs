<!--  ⚡️ Give your PyTorch models superpowers ⚡️ -->

<div align="center">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
  <img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Lightning Thunder: Supercharge Your PyTorch Models

**Lightning Thunder** is a source-to-source compiler that empowers you to optimize your PyTorch models with ease, delivering significant performance gains and unlocking the full potential of your hardware.  [Explore the Lightning Thunder GitHub repository](https://github.com/Lightning-AI/lightning-thunder) to get started.

**Key Features:**

*   🚀 **Accelerated Training and Inference:** Experience up to 40% faster PyTorch model execution.
*   ⚙️ **Composable Optimizations:** Easily integrate custom kernels, fusions, quantization, and distributed strategies.
*   💡 **Simplified Optimization:**  Designed for both beginners and expert performance engineers.
*   🎯 **Pre-built Plugins:** Leverage ready-to-use plugins for immediate speed-ups, including CUDA Graphs and more.
*   🔬 **Flexible Precision:**  Supports FP4/FP6/FP8 precision for optimal performance.
*   🌐 **Distributed Training:**  Integrates TP/PP/DP distributed strategies.
*   🧠 **LLM and non-LLM Support:** Works on a wide range of models and use cases.
*   🛠️ **Custom Kernel Integration:**  Seamlessly integrate custom Triton kernels.
*   💡 **Blackwell Support:** Ready for next-generation NVIDIA Blackwell hardware.

<div align='center'>

<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>

</div>

---

### Quick Start

Get started with Lightning Thunder by following these steps:

**1. Installation:**

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

*(See [Installation Documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for more advanced installation options.)*

**2. Basic Usage:**

```python
import torch
import torch.nn as nn
import thunder

# Define your PyTorch model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Optimize the model with Thunder
thunder_model = thunder.compile(model)

# Prepare input data
x = torch.randn(64, 2048)

# Run the optimized model
y = thunder_model(x)

# Verify the output (optional)
torch.testing.assert_close(y, model(x))
```

### Examples

Lightning Thunder offers easy-to-use examples to get you up and running.  Here are some examples:

*   [Speed up LLM training](https://github.com/Lightning-AI/lightning-thunder#speed-up-llm-training)
*   [Speed up HuggingFace BERT inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-bert-inference)
*   [Speed up HuggingFace DeepSeek R1 distill inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-deepseek-r1-distill-inference)
*   [Speed up Vision Transformer inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-vision-transformer-inference)
*   [Benchmarking HF models](https://github.com/Lightning-AI/lightning-thunder#benchmarking-hf-models)

### Plugins

Enhance your model's performance with Thunder's plugin system.

*   **Distributed Strategies:** Scale with DDP, FSDP, TP.
*   **Numerical Precision:** Optimize with FP8 and MXFP8.
*   **Quantization:** Reduce memory usage.
*   **CUDA Graphs:** Reduce CPU overhead.
*   **Debugging and Profiling**

**Example: Reduce CPU Overheads with CUDA Graphs**
```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

### How it Works

Thunder uses a three-stage process:

1.  **Acquire:** Interprets Python bytecode.
2.  **Transform:** Optimizes the computation trace.
3.  **Execute:** Routes parts for execution (fusion, libraries, kernels, PyTorch eager operations).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

### Performance

Achieve significant speedups in your pre-training tasks.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

### Community

Join the Thunder community:

*   💬 [Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)