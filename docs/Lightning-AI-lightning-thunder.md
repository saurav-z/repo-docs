<!--
<div align='center'>
-->

# Lightning Thunder: Supercharge Your PyTorch Models for Blazing-Fast Performance ⚡️

**Lightning Thunder** is a source-to-source compiler for PyTorch that drastically accelerates model training and inference, making it easier than ever to optimize your AI models. [Learn more on GitHub](https://github.com/Lightning-AI/lightning-thunder).

<!--
</div>

<div align="center">
-->
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
<br/>
<br/>

<!--
&#160;

<strong>Source-to-source compiler for PyTorch.</strong>
Fast. Understandable. Extensible.
-->

## Key Features

*   🚀 **Blazing Speed:** Achieve up to 81% faster performance in LLM inference and significant speedups in training.
*   ⚙️ **Easy Optimization:** Seamlessly integrate custom kernels, fusions, quantization, and distributed strategies.
*   📦 **Out-of-the-Box Performance:** Utilize pre-built plugins for instant model speed-ups.
*   🔬 **Extensible Framework:** Fine-tune and optimize your AI models with a composable transformation framework.
*   ✅ **Comprehensive Support:** Benefit from quantization, FP4/FP6/FP8 precision, distributed training/inference (TP/PP/DP), CUDA Graphs, and more.
*   🛠️ **Flexible for All Users:** Offers solutions for both end-users and expert performance engineers.
*   🔥 **Ready for Next-Gen Hardware:** Designed to take advantage of NVIDIA Blackwell and other cutting-edge hardware.

<!--
</div>

<div align='center'>
-->
<pre>
✅ Run PyTorch 40% faster   ✅ Quantization                ✅ Kernel fusion
✅ Training recipes         ✅ FP4/FP6/FP8 precision       ✅ Distributed TP/PP/DP
✅ Inference recipes        ✅ Ready for NVIDIA Blackwell  ✅ CUDA Graphs
✅ LLMs, non LLMs and more  ✅ Custom Triton kernels       ✅ Compose all the above
</pre>
<!--
</div>
-->

<div align="center">
    <img src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

Get started in minutes!

1.  **Install Thunder:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```

    *   For advanced installation options, including Blackwell support and optional executors, see the [original README](https://github.com/Lightning-AI/lightning-thunder).

2.  **Optimize Your Model:**

    ```python
    import thunder
    import torch.nn as nn
    import torch

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

## Examples

Thunder easily accelerates various models with example scripts.

*   **Speed up LLM training:** Run LitGPT with significant performance gains.
*   **Speed up HuggingFace BERT inference:** Integrate with transformers for faster inference.
*   **Speed up HuggingFace DeepSeek R1 distill inference:** Leverage transformers for increased efficiency.
*   **Speed up Vision Transformer inference:** Optimize vision models with Thunder.

    Run the following command in the terminal:
    ```bash
    python examples/quickstart/hf_llm.py
    ```
    to see the difference in speed.

## Plugins

Thunder offers a range of plugins for customizable optimizations:

*   **Distributed Strategies:** DDP, FSDP, TP
*   **Numerical Precision:** FP8, MXFP8
*   **Quantization:** Save memory
*   **CUDA Graphs:** Reduce latency
*   **Debugging and Profiling**

Example to reduce CPU overhead via CUDAGraphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Thunder transforms your PyTorch models through these stages:

1.  **Acquire:** Interprets Python bytecode and creates a straight-line program.
2.  **Transform:** Modifies the computation trace for optimizations.
3.  **Execute:** Routes parts of the trace for efficient execution using fusion, specialized libraries, custom kernels, and eager PyTorch operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers impressive speed-ups, as shown in pre-training tasks.

*   Significant speedups on H100 and B200 hardware.

## Community

Lightning Thunder is an open-source project, inviting collaboration.

*   💬 [Join the Discord community](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)

```