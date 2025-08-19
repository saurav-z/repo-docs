<!-- Improved & Summarized README - SEO Optimized -->

# Lightning Thunder: Supercharge Your PyTorch Models ⚡️

**Lightning Thunder is a source-to-source compiler that drastically accelerates your PyTorch models, empowering you with cutting-edge performance and optimization capabilities.** Learn more about Lightning Thunder on the [original repo](https://github.com/Lightning-AI/lightning-thunder).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   ✅ **Up to 8x Faster Training and Inference:** Experience dramatic speed improvements for your PyTorch models.
*   ✅ **Simplified Optimization:** Easily integrate custom kernels, fusion techniques, quantization, and distributed strategies.
*   ✅ **FP4/FP6/FP8 Precision:** Optimize numerical precision for enhanced efficiency.
*   ✅ **Broad Compatibility:** Supports Large Language Models (LLMs) and non-LLMs.
*   ✅ **Ready for Next-Gen Hardware:** Optimized for NVIDIA Blackwell and other cutting-edge hardware.
*   ✅ **Composable Transformations:** Build, modify, and optimize AI models with ease.
*   ✅ **CUDA Graphs:** Reduce CPU overhead and improve performance.
*   ✅ **Custom Kernels:** Extend functionality with custom Triton kernels.
*   ✅ **Distributed Training:** Support for Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel (PP) strategies.
*   ✅ **Pre-Built Plugins:** Access pre-configured plugins for common optimization tasks.

<div align='center'>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)

</div>

<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick start</a> •
    <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <!-- <a target="_blank" href="#hosting-options" style="margin: 0 10px;">Hosting</a> • -->
    <a target="_blank" href="https://lightning.ai/docs/thunder/latest/" style="margin: 0 10px;">Docs</a>
  </div>
</div>

&#160;

<!--
<div align="center">
<a target="_blank" href="https://lightning.ai/docs/thunder/home/get-started">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>
</div>
-->

&#160;

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Quick Start

1.  **Install:**

    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
    **(For advanced install options including Blackwell support, see the original README)**

2.  **Integrate:**

    ```python
    import thunder
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

## Examples

*   **Speed up LLM Training** See examples using LitGPT
*   **Speed up Hugging Face BERT Inference**
*   **Speed up Hugging Face DeepSeek R1 Distill Inference**
*   **Speed up Vision Transformer Inference**
*   **Benchmarking HF models**

(Refer to the original README for full example code and instructions.)

## Plugins

Thunder provides a range of plugins for easily applying optimizations, including:

*   **Distributed Strategies:** Scale up with DDP, FSDP, TP.
*   **Numerical Precision:** Optimize with FP8, MXFP8.
*   **Quantization:** Reduce memory usage.
*   **CUDAGraphs:** Reduce CPU overhead.
*   **Debugging and Profiling:** Identify bottlenecks.

**Example: Using CUDAGraphs**

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How it Works

Thunder employs a three-stage process:

1.  **Acquisition:** Interprets Python bytecode.
2.  **Transformation:** Transforms the computation trace for optimization.
3.  **Execution:** Routes parts of the trace for efficient execution (fusion, specialized libraries, custom kernels, PyTorch eager operations).

## Performance

Lightning Thunder delivers significant performance gains, as illustrated by the speed-ups achieved on pre-training tasks using LitGPT on H100 and B200 hardware.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Thunder is an open-source project developed in collaboration with the community.

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)