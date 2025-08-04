# Lightning Thunder: Supercharge Your PyTorch Models

**Lightning Thunder is a source-to-source compiler that accelerates PyTorch models, offering significant speedups with minimal code changes.**

[Explore the Lightning Thunder repository](https://github.com/Lightning-AI/lightning-thunder).

---

## Key Features

*   ⚡ **Blazing Fast Performance:** Achieve up to 81% faster training and inference for various model architectures.
*   ⚙️ **Optimized for Modern Hardware:**  Leverages NVIDIA Blackwell, CUDA Graphs, and other cutting-edge technologies.
*   🚀 **Model Optimization Plugins:** Pre-built plugins for quantization, distributed training (TP/PP/DP), kernel fusion, and more.
*   🧠 **Extensible & Customizable:** Designed for performance experts to understand, modify, and optimize AI models using composable transformations.
*   🛠️ **Easy to Use:**  Simple `thunder.compile()` API for effortless integration and speedups.
*   🧪 **Ready for LLMs & Beyond:** Supports LLMs, non-LLMs, and custom Triton kernels.

---

## Why Use Lightning Thunder?

Lightning Thunder provides an easy-to-use framework for optimizing PyTorch models, offering significant performance improvements with minimal code changes.  Whether you're a performance expert or a user looking for out-of-the-box speedups, Thunder has you covered.

---

## Getting Started

### Installation

Install Lightning Thunder using pip:

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Advanced Install Options:**
*   **Blackwell Support:**
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com
    pip install lightning-thunder
    ```

*   **Additional Executors:**
    ```bash
    pip install nvidia-cudnn-frontend  # cuDNN SDPA
    pip install "transformer_engine[pytorch]" # Float8 support
    ```

*   **Bleeding Edge:**
    ```bash
    pip install git+https://github.com/Lightning-AI/lightning-thunder.git@main
    ```

*   **Development:**
    ```bash
    git clone https://github.com/Lightning-AI/lightning-thunder.git
    cd lightning-thunder
    pip install -e .
    ```

### Hello World Example

```python
import torch
import torch.nn as nn
import thunder

# Define a simple model
model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))

# Compile the model with Thunder
thunder_model = thunder.compile(model)

# Generate input data
x = torch.randn(64, 2048)

# Run inference
y = thunder_model(x)

# Verify results
torch.testing.assert_close(y, model(x))
```

---

## Examples

*   **Speed up LLM Training:** Integrates seamlessly with LitGPT for faster training.
*   **Accelerate Hugging Face BERT Inference:** Supports various Hugging Face models.
*   **Speed up Vision Transformer Inference:** Optimize vision models with Thunder.
*   **HuggingFace DeepSeek R1 distill Inference** Use Thunder to accelerate DeepSeek models.

## Performance

Thunder delivers significant speedups.  See the example pre-training task results using LitGPT on H100 and B200 hardware:

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Community & Resources

*   💬 [Join the Discord community for help and discussions](https://discord.com/invite/XncpTy7DSt)
*   📄 [Read the full documentation](https://lightning-thunder.readthedocs.io/en/latest/)
*   📝 [View the License (Apache 2.0)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)