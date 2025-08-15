# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder is a source-to-source compiler that unlocks significant performance improvements for your PyTorch models.** [Explore the Lightning Thunder repository](https://github.com/Lightning-AI/lightning-thunder).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>
</div>

Lightning Thunder empowers both **end-users** and **performance experts** with:

*   **Up to 40% Faster PyTorch Model Execution**: Experience significant speedups out-of-the-box.
*   **Optimized Precision**: Leverage FP8, FP6, and FP4 for efficient model execution.
*   **Flexible Distributed Training**: Utilize TP/PP/DP strategies for scaling.
*   **Advanced Kernel Fusion**: Benefit from optimized kernel fusion techniques.
*   **Custom Kernel Integration**: Extend performance with custom Triton kernels.
*   **LLM and Non-LLM Support**: Accelerate a wide range of model architectures.
*   **CUDA Graphs Integration**: Reduce CPU overheads for enhanced performance.
*   **Ready for NVIDIA Blackwell**: Fully optimized to leverage the latest hardware.

<div align='center'>
<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>
</div>

## Quick Start

Get started with Lightning Thunder:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

For advanced installation options, refer to the [official documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

### Hello World Example

Optimize your PyTorch models with Thunder:

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

## Key Features

*   **Source-to-Source Compilation**:  Transforms PyTorch code for optimized execution.
*   **Plugin-Based Architecture**: Easily apply various optimizations like quantization and distributed training.
*   **CUDA Graph Integration**: Reduce CPU overhead for performance gains.
*   **Broad Hardware Support**:  Optimized for NVIDIA GPUs, including Blackwell.
*   **Performance Focused**: Designed for speed and ease of optimization.

## Examples

Lightning Thunder provides significant speedups for various models:

*   **Speed up LLM training**  See example code [here](https://github.com/Lightning-AI/lightning-thunder#speed-up-llm-training).
*   **Speed up HuggingFace BERT inference** See example code [here](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-bert-inference).
*   **Speed up HuggingFace DeepSeek R1 distill inference** See example code [here](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-deepseek-r1-distill-inference).
*   **Speed up Vision Transformer inference** See example code [here](https://github.com/Lightning-AI/lightning-thunder#speed-up-vision-transformer-inference).

Run `python examples/quickstart/hf_llm.py` to see speed improvements:  Thunder can be up to 81% faster.

## Performance

Lightning Thunder offers significant performance improvements, as demonstrated by pre-training tasks on H100 and B200 hardware:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)