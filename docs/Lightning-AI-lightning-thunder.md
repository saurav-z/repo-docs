# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder accelerates your PyTorch models with custom kernels, quantization, and more, delivering significant performance boosts for both experts and end-users. [Explore the original repo](https://github.com/Lightning-AI/lightning-thunder)**

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Blazing Fast:** Achieve up to 40% faster PyTorch model execution with minimal code changes.
*   **End-to-End Optimization:** Leverage a suite of optimizations, including quantization (FP4/FP6/FP8), kernel fusion, and distributed training strategies (TP/PP/DP).
*   **Seamless Integration:** Utilize pre-built plugins for out-of-the-box performance gains, or create custom transformations for advanced users.
*   **LLM Ready:** Optimized for Large Language Models, non LLMs, and more.
*   **Hardware-Aware:** Optimized for latest-gen NVIDIA hardware, including Blackwell.
*   **Extensible:** Supports custom Triton kernels, CUDA Graphs, and composable transformations.

<div align='center'>
<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>
</div>

## Quick Start

### Installation

Install Thunder using pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Advanced Installation Options:**
*   **Blackwell Support:** [See Installation instructions in the original README](https://github.com/Lightning-AI/lightning-thunder#blackwell-support)
*   **Additional Executors:** [See Installation instructions in the original README](https://github.com/Lightning-AI/lightning-thunder#install-additional-executors)
*   **Bleeding Edge and Development Installations:** [See Installation instructions in the original README](https://github.com/Lightning-AI/lightning-thunder#install-thunder-bleeding-edge)

### Hello World Example

Optimize your PyTorch models with a single line of code:

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

**Speed up your models in a few lines of code!**

*   **LLM Training:** Leverage LitGPT and achieve faster training times. [See the example](https://github.com/Lightning-AI/lightning-thunder#speed-up-llm-training)
*   **Hugging Face BERT Inference:** Accelerate inference for your favorite transformer models.  [See the example](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-bert-inference)
*   **Hugging Face DeepSeek R1 Inference:** Achieve increased inference performance for DeepSeek R1 models.  [See the example](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-deepseek-r1-distill-inference)
*   **Vision Transformer Inference:** Optimize vision models with Thunder. [See the example](https://github.com/Lightning-AI/lightning-thunder#speed-up-vision-transformer-inference)
*   **Benchmarking HF Models:**  [See the example](https://github.com/Lightning-AI/lightning-thunder#benchmarking-hf-models)

## Plugins

Thunder plugins enable a wide array of model optimizations with minimal code changes.  Plugins can be enabled by passing the plugin name to the `thunder.compile` function.

*   **Distributed Training:** Scale with DDP, FSDP, and TP.
*   **Precision Optimization:**  FP8, MXFP8 support.
*   **Memory Savings:**  Quantization for reduced memory footprint.
*   **Latency Reduction:**  Leverage CUDA Graphs.
*   **Debugging & Profiling:** Tools for understanding and optimizing your models.

## How It Works

Thunder employs a three-stage process to optimize your models:

1.  **Trace Acquisition:**  Interprets Python bytecode to generate a straight-line program representation.
2.  **Transformation:**  Transforms the computation trace for distribution, precision changes, and more.
3.  **Execution Routing:** Routes parts of the trace to various execution engines for optimized performance.

    *   fusion (NVFuser, torch.compile)
    *   specialized libraries (e.g. cuDNN SDPA, TransformerEngine)
    *   custom Triton and CUDA kernels
    *   PyTorch eager operations

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers significant performance improvements.  See the performance graphs in the original README for pre-training task results using LitGPT on H100 and B200 hardware.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Thunder is a community-driven open-source project.

*   💬 [Join the Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)