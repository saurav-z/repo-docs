# Lightning Thunder: Supercharge Your PyTorch Models ⚡

Lightning Thunder is a source-to-source compiler for PyTorch that unlocks significant performance gains and simplifies model optimization.  [Explore the Lightning Thunder Repository](https://github.com/Lightning-AI/lightning-thunder).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Accelerated Performance:** Achieve up to 40% faster PyTorch model execution, with benchmarks showing significant speedups for LLMs, vision transformers, and other models.
*   **Easy Optimization:**  Simplify model optimization with custom kernels, fusions, quantization, distributed strategies, and more.
*   **Composable Transformations:**  A flexible framework for understanding, modifying, and optimizing AI models through composable transformations.
*   **Out-of-the-box Plugins:** Utilize pre-built plugins for immediate performance improvements, including distributed strategies, quantization, and CUDA Graph integration.
*   **FP8 Support:** Optimized support for FP8 precision, allowing for more efficient computations
*   **Broad Compatibility:** Ready for NVIDIA Blackwell and other cutting-edge hardware.
*   **Extensive Examples:** Includes a range of examples for accelerating LLM training and inference (LitGPT, Hugging Face models), vision transformers, and more.

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

1.  **Install Required Packages:**
    ```bash
    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
    pip install lightning-thunder
    ```
    For more installation options including support for Blackwell and bleeding edge versions, refer to the [installation instructions](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

2.  **Import Thunder and Define Your Model:**

    ```python
    import thunder
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    ```

3.  **Compile and Run:**

    ```python
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

## Examples

*   [Speed up LLM training](https://github.com/Lightning-AI/lightning-thunder#speed-up-llm-training)
*   [Speed up HuggingFace BERT inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-bert-inference)
*   [Speed up HuggingFace DeepSeek R1 distill inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-huggingface-deepseek-r1-distill-inference)
*   [Speed up Vision Transformer inference](https://github.com/Lightning-AI/lightning-thunder#speed-up-vision-transformer-inference)
*   [Benchmarking HF models](https://github.com/Lightning-AI/lightning-thunder#benchmarking-hf-models)

## Plugins

Extend Thunder's capabilities with plugins to:

*   Scale up models using distributed strategies (DDP, FSDP, TP).
*   Optimize numerical precision with FP8 and MXFP8.
*   Reduce memory usage via quantization.
*   Reduce latency with CUDA Graphs.

    Apply CUDA Graphs optimization:
    ```python
    thunder_model = thunder.compile(model, plugins="reduce-overhead")
    ```

## How It Works

Thunder accelerates PyTorch models through a three-stage process:

1.  **Acquisition:** Interprets your model's bytecode to create a straight-line Python program.
2.  **Transformation:** Modifies the computation trace for distribution and precision changes.
3.  **Execution Routing:** Executes parts of the trace using optimized methods, including fusion, specialized libraries (cuDNN SDPA, TransformerEngine), custom kernels (Triton, CUDA), and standard PyTorch operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Join the Lightning Thunder community:

*   💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)