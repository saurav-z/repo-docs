<!-- ALL-IN-ONE SEO OPTIMIZED README -->

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder is a source-to-source compiler that unlocks unparalleled performance for your PyTorch models.**  Boost your models with custom kernels, fusions, quantization, distributed strategies, and more, all while maintaining readability and extensibility. **[Check out the Lightning Thunder repo!](https://github.com/Lightning-AI/lightning-thunder)**

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
</div>

## Key Features

*   🚀 **Significant Speed-Ups:** Achieve up to 40% faster model execution out-of-the-box with optimized kernels and plugins.
*   ✅ **Easy Optimization:** Effortlessly integrate advanced techniques like quantization (FP4/FP6/FP8), kernel fusion, and distributed training (TP/PP/DP).
*   🛠️ **Extensible Framework:**  Provides a flexible and composable architecture for customizing and optimizing AI models.
*   💡 **Performance Experts' Toolkit:**  The go-to framework for understanding, modifying, and optimizing AI models.
*   💻 **Broad Hardware Support:** Ready for NVIDIA Blackwell and other last-generation hardware, maximizing performance.
*   🧩 **Built-in Plugins:** Leverage pre-built plugins for CUDAGraphs, FP8, and more.
*   🧠 **Supports Diverse Models:** Compatible with LLMs, non-LLMs, and a wide range of PyTorch models.

<div align='center'>
<pre>
✅ Run PyTorch models faster ✅ Quantization (FP4/FP6/FP8)   ✅ Kernel Fusion
✅ Training recipes        ✅ Distributed TP/PP/DP      ✅ CUDA Graphs
✅ LLMs and other Models   ✅ Custom Triton kernels     ✅ Compose all the above
</pre>
</div>

## Quick Start

Get started with Thunder in just a few lines of code:

1.  **Install Thunder:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
2.  **Optimize Your Model:**

    ```python
    import thunder
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

    For detailed installation instructions and advanced options, refer to the [Installation Guide](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) in our docs.

## Examples

Explore real-world speedups with Lightning Thunder:

*   **LLM Training:** Train LLMs like Llama 3.2x faster. See how to speed up LitGPT training in the [examples section](https://github.com/Lightning-AI/lightning-thunder#examples).
*   **Hugging Face BERT Inference:** Accelerate inference for models like BERT.
*   **Vision Transformer Inference:** Optimize computer vision models such as ViT.

## Performance Highlights

Thunder delivers substantial performance gains.
<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>
## How Thunder Works

Thunder's process for optimizing PyTorch models:
<ol>
<li> **Trace Acquisition**:  Interprets Python bytecode to create a straight-line Python program representation of the model.</li>
<li> **Computation Transformations**: Modifies the computation trace for distribution and to alter precision.</li>
<li> **Execution Routing**: Executes parts of the computation trace with:
   <ul>
   <li> Fusion (NVFuser, `torch.compile`)</li>
   <li> Specialized libraries (cuDNN SDPA, TransformerEngine)</li>
   <li> Custom Triton and CUDA kernels</li>
   <li> PyTorch eager operations</li>
   </ul>
</li>
</ol>
<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Community and Resources

*   💬 **Get Help:** [Join our Discord](https://discord.com/invite/XncpTy7DSt)
*   📚 **Documentation:** [Read the Docs](https://lightning.ai/docs/thunder/latest/)
*   📜 **License:** [Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)