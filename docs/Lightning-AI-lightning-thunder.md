# Lightning Thunder: Supercharge Your PyTorch Models ⚡️

Lightning Thunder is a powerful source-to-source compiler that empowers you to optimize and accelerate your PyTorch models with ease.  **Get up to 80% faster model performance with Thunder!**  [Learn more at the Lightning Thunder GitHub Repository](https://github.com/Lightning-AI/lightning-thunder).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Fast Performance:** Achieve significant speedups (up to 80%) compared to standard PyTorch.
*   **Easy Optimization:**  Optimize models with plugins for various strategies.
*   **Model Customization:** Easily augment your models with custom kernels, fusions, quantization, and distributed strategies.
*   **FP4/FP6/FP8 Support:** Optimize numerical precision for improved performance and memory usage.
*   **Distributed Training:** Supports TP/PP/DP distributed training strategies.
*   **Hardware Ready:** Designed for optimal utilization of NVIDIA Blackwell and CUDA Graphs.
*   **Extensive Compatibility:** Works seamlessly with LLMs, non-LLMs, and custom Triton kernels.
*   **Composable Transformations:** Performance experts can modify and optimize AI models through composable transformations.

<div align='center'>
    <pre>
    ✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
    ✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
    ✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
    ✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
    </pre>
</div>

### Quick Start

Install Thunder:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

*For Blackwell support, install CUDA 12.8 and use the nightly PyTorch builds.*  See the original README for full installation details.

```python
import torch.nn as nn
import torch
import thunder

# Define a simple model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile your model with Thunder
thunder_model = thunder.compile(model)

# Run your model
x = torch.randn(64, 2048)
y = thunder_model(x)

torch.testing.assert_close(y, model(x))
```

## Examples

Here are a few examples of how to speed up some models:

*   **Speed up LLM training**: Install LitGPT and then run a few lines of code.
*   **Speed up HuggingFace BERT inference**: Import transformers and then run your inference through Thunder.
*   **Speed up HuggingFace DeepSeek R1 distill inference**: Import transformers and then run your inference through Thunder.
*   **Speed up Vision Transformer inference**: Use the vision transformer, load it into Thunder, and then run your inference.

## Plugins

Thunder plugins provide a flexible way to apply various optimizations.
Some available plugins are:

*   **Reduce CPU overhead with CUDAGraphs**: `thunder_model = thunder.compile(model, plugins="reduce-overhead")`
*   **Scale up with distributed strategies** with DDP, FSDP, TP.
*   **Optimize numerical precision** with FP8, MXFP8.
*   **Save memory with quantization**.
*   **Debugging and profiling**.

## How it Works

Thunder transforms your PyTorch models in three key stages:

1.  **Acquisition:** Interprets Python bytecode and produces a straight-line Python program.
2.  **Transformation:** Transforms the computation trace for distribution and precision adjustments.
3.  **Execution:** Routes parts of the trace for optimized execution using fusions, specialized libraries, custom kernels, and eager PyTorch operations.

## Performance

See the performance graph below comparing Lightning Thunder against PyTorch eager, showcasing the significant speedups achieved on various hardware configurations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Thunder is an open-source project, developed with significant community contributions.

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)