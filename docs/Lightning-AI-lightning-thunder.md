<!--
  _  __  _    _  _____  _     _  _____   _       _   _
 | |/ / | |  | ||  _  || |   | ||  ___| | |     | | | |
 | ' /  | |  | || | | || |   | || |_    | |     | | | |
 |  <   | |  | || | | || |   | ||  _|   | |     | | | |
 | . \  | |_ | || |_| || |___| || |_    | |___  | |_| |
 |_|\_\ \___/ ||_____||_____|_||_____| \_____/  \___/
-->

# Lightning Thunder: Supercharge Your PyTorch Models

⚡ **Lightning Thunder accelerates your PyTorch models with a source-to-source compiler, unlocking performance gains and advanced optimization capabilities.** Discover how Thunder can transform your models with speed and efficiency! ([View on GitHub](https://github.com/Lightning-AI/lightning-thunder))

<div align="center">
  <img src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" alt="Thunder Light Mode" />
  <img src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" alt="Thunder Dark Mode" />
</div>

## Key Features

*   🚀 **Blazing-Fast Performance:** Achieve up to 81% faster model execution with optimized kernels and compilation techniques.
*   💡 **Easy Optimization:** Enhance models with custom kernels, fusions, quantization, and distributed strategies.
*   🛠️ **Composable Transformations:** Leverage a flexible framework for understanding, modifying, and optimizing AI models.
*   🔬 **Advanced Precision:** Explore FP4/FP6/FP8 precision for memory and performance benefits.
*   🌐 **Distributed Training and Inference:** Utilize distributed training strategies (TP/PP/DP) for scaling up model training.
*   🔥 **Cutting-Edge Support:** Ready for NVIDIA Blackwell and other latest hardware.
*   ⚙️ **Customizable with Plugins:** Extend functionality with pre-built and custom plugins for CUDA Graphs, LLMs, and more.
*   ✅ **Pre-built recipes:** Easily apply speedups for LLMs, Vision Transformers and more

<div align='center'>

<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion        
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP 
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs          
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>

</div>

## Quick Start

Get started with Lightning Thunder in a few simple steps:

1.  **Installation:**
    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
    *For Blackwell, see the original README for advanced installation options.*

2.  **Basic Example:**

    ```python
    import torch
    import torch.nn as nn
    import thunder

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    assert torch.testing.assert_close(y, model(x))
    ```

## Examples

Explore how Lightning Thunder can be used to speed up a variety of models:

*   **Speed up LLM training**: Integrate with LitGPT.
*   **Speed up Hugging Face BERT inference:** Use with Transformers.
*   **Speed up Hugging Face DeepSeek R1 distill inference:** Use with Transformers
*   **Speed up Vision Transformer inference:** Integrate with Torchvision.

## Performance

Lightning Thunder delivers impressive performance gains.  Here's a look at speed-ups on pre-training tasks using LitGPT on H100 and B200 hardware, relative to PyTorch eager:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community for support and collaboration:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)