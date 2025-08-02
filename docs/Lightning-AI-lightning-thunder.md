# Lightning Thunder: Supercharge Your PyTorch Models ⚡

Lightning Thunder is a source-to-source compiler for PyTorch that makes optimizing your models easy, fast, and understandable.  [Learn more on GitHub](https://github.com/Lightning-AI/lightning-thunder).

**Key Features:**

*   **Accelerated Training and Inference:** Achieve significant speedups (up to 40% faster) with minimal code changes.
*   **Model Optimization:** Leverage custom kernels, fusions, quantization, and distributed strategies for optimal performance.
*   **Ease of Use:** Comes with out-of-the-box plugins for immediate speed improvements.
*   **Flexibility:** Adaptable for both end-users and performance experts, allowing for composable model transformations.
*   **Broad Compatibility:** Supports LLMs, non-LLMs, and various hardware, including NVIDIA Blackwell.
*   **Advanced Techniques:** Includes support for FP4/FP6/FP8 precision, CUDA Graphs, and custom Triton kernels.
*   **Distributed Training:**  Supports distributed training strategies like TP/PP/DP.

## Get Started

### Installation

Install Thunder via pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Note:**  For more advanced installation options, including Blackwell support and bleeding-edge versions, please see the [full installation instructions](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html).

### Example:  Hello World

```python
import torch.nn as nn
import thunder
import torch

# Define your model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Optimize with Thunder
thunder_model = thunder.compile(model)

# Prepare input data
x = torch.randn(64, 2048)

# Run the model
y = thunder_model(x)

# Verify the output
torch.testing.assert_close(y, model(x))
```

## Examples and Benchmarks

Thunder provides pre-built examples to demonstrate performance improvements across various model types, including:

*   Speed up LLM training
*   Speed up HuggingFace BERT inference
*   Speed up HuggingFace DeepSeek R1 distill inference
*   Speed up Vision Transformer inference

### Benchmarking HF models

Thunder can achieve the following speedups:

*   **Text generation:** 3.36x - 3.42x faster
*   **Forward pass:** 1.51x - 1.63x faster
*   **Forward pass + loss:** 1.55x - 1.64x faster
*   **Forward + backward:** 1.51x - 1.69x faster

For more information on benchmarks, run the following code:
```bash
python examples/quickstart/hf_llm.py
```

## Plugins for Optimization

Thunder utilizes plugins to easily apply optimizations. Examples:

*   **Distributed strategies:** DDP, FSDP, TP
*   **Precision optimization:** FP8, MXFP8
*   **Memory optimization:** Quantization
*   **Latency reduction:** CUDAGraphs
*   **Debugging and profiling**

For example, to reduce CPU overhead using CUDAGraphs:

```python
thunder_model = thunder.compile(model, plugins="reduce-overhead")
```

## How Thunder Works

Thunder optimizes PyTorch models through three key stages:

1.  **Acquisition:**  Interprets Python bytecode and creates a straight-line Python program.
2.  **Transformation:**  Transforms the computation trace for distribution and precision changes.
3.  **Execution Routing:** Routes the trace for execution, using fusion, specialized libraries, custom kernels, and eager operations.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/how_it_works.png" width="800px" style="max-width: 100%;">
</div>

## Performance

Thunder delivers significant performance gains, as shown in the pre-training task using LitGPT on H100 and B200 hardware.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community

Lightning Thunder is open source and developed with community contributions.

*   💬 [Join the Discord](https://discord.com/invite/XncpTy7DSt)
*   📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)