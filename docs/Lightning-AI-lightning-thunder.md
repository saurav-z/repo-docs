# Lightning Thunder: Supercharge Your PyTorch Models ⚡

[Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) is a source-to-source compiler for PyTorch that unlocks unprecedented performance gains and optimization capabilities for your AI models.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

**Key Features:**

*   **Blazing Fast Performance:** Achieve up to 81% speedups on LLMs and other models.
*   **Effortless Optimization:** Easy-to-use plugins for quantization, kernel fusion, and distributed training.
*   **Flexible Precision:** Supports FP4/FP6/FP8, and mixed-precision training.
*   **Distributed Training Support:** Utilize TP/PP/DP strategies.
*   **Hardware Ready:** Optimized for NVIDIA Blackwell and other latest-generation hardware.
*   **Customization:** Extendable with custom Triton kernels.
*   **Composable:** Combine optimizations for maximum impact.

## Quick Start

Get up and running with Lightning Thunder in minutes:

1.  **Install:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
    *(See [installation documentation](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html) for detailed installation options)*
2.  **Integrate:**

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

## Examples

*   **Speed up LLM training**:  Lightning Thunder offers significant speedups. ([See examples](https://github.com/Lightning-AI/lightning-thunder#examples))
*   **Speed up HuggingFace BERT Inference**
*   **Speed up HuggingFace DeepSeek R1 Distill Inference**
*   **Speed up Vision Transformer inference**

## Plugins

Thunder offers a flexible plugin system for applying various optimizations:

*   Distributed Training (DDP, FSDP, TP)
*   Numerical Precision Optimization (FP8, MXFP8)
*   Memory Savings with Quantization
*   Reduced Latency with CUDA Graphs
*   Debugging and Profiling

    ```python
    thunder_model = thunder.compile(model, plugins="reduce-overhead")
    ```

## How it Works

Thunder transforms your PyTorch models in three core stages:

1.  **Acquisition:** Interprets Python bytecode to create a straight-line Python program.
2.  **Transformation:** Optimizes the computation trace through distribution and precision adjustments.
3.  **Execution:** Routes parts of the trace through fused kernels, specialized libraries, custom kernels, and eager PyTorch operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Lightning Thunder offers significant speedups on a wide range of hardware and tasks:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community!

*   💬 [Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)