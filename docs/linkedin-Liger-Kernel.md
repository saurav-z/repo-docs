# Liger Kernel: Accelerate LLM Training with Efficient Triton Kernels

Supercharge your Large Language Model (LLM) training with **Liger Kernel**, a collection of optimized Triton kernels designed to significantly boost performance and reduce memory usage.  Get the source code and join the community on [GitHub](https://github.com/linkedin/Liger-Kernel)!

---

## Key Features

*   🚀 **Enhanced Performance:** Increase multi-GPU training throughput by up to **20%**.
*   💾 **Reduced Memory Footprint:** Lower memory usage by up to **60%**, enabling larger models and longer context lengths.
*   💡 **Easy Integration:**  Patch Hugging Face models with a single line of code, or compose your own models with our modular kernels.
*   🔬 **Accurate & Reliable:** Exact computation without approximations, ensuring training accuracy. Rigorous unit tests and convergence testing.
*   📦 **Lightweight Dependencies:** Minimal dependencies - only PyTorch and Triton are required.
*   ⚙️ **Multi-GPU Support:** Compatible with various multi-GPU setups (PyTorch FSDP, DeepSpeed, DDP, etc.).
*   ✅ **Framework Integration:**  Works seamlessly with popular training frameworks like Axolotl, LLaMA-Factory, Hugging Face Trainer, and others.
*   🧠 **Optimized Post-Training:** Includes optimized kernels for alignment and distillation tasks, offering up to 80% memory savings.
*   🏢 **Community Driven:**  We welcome contributions from the community to develop the best kernels for LLM training.

---

## Installation

Liger Kernel offers both stable and nightly builds. Follow the installation instructions below to get started.

### Prerequisites

*   **CUDA:**
    *   `torch >= 2.1.2`
    *   `triton >= 2.3.0`
*   **ROCm:**
    *   `torch >= 2.5.0`
    *   `triton >= 3.0.0`

    ```bash
    # CUDA install stable version:
    pip install liger-kernel
    # CUDA install nightly version:
    pip install liger-kernel-nightly

    # ROCm installation from source:
    git clone https://github.com/linkedin/Liger-Kernel.git
    cd Liger-Kernel
    pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
    ```

### Optional Dependencies
*   `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.

---

## Getting Started

Liger Kernel can be integrated into your LLM training workflow through:

1.  **AutoModel Wrapper:**  The easiest way to apply the optimized kernels.

    ```python
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained("path/to/your/model")
    ```

2.  **Model-Specific Patching:** Apply optimized kernels for specific models.

    ```python
    import transformers
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama() # Apply to Llama models
    model = transformers.AutoModelForCausalLM.from_pretrained("path/to/llama/model")
    ```

3.  **Modular Kernels:**  Compose your own models using individual kernels.

    ```python
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import torch.nn as nn
    import torch

    model = nn.Linear(128, 256).cuda()
    loss_fn = LigerFusedLinearCrossEntropyLoss()
    # ... rest of the code
    ```

---

## Performance & Benchmarks

Liger Kernel offers significant performance gains and memory reduction.  The banner below showcases these improvements. For specific benchmarks, see [the original README](https://github.com/linkedin/Liger-Kernel).

![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)

---

## Available Kernels

A list of the High-level and low-level APIs are available in the original README, accessible via [the original README](https://github.com/linkedin/Liger-Kernel).
The main categories are:
* High-level APIs:
    *   AutoModel
    *   Patching
* Low-level APIs:
    *   Model Kernels
    *   Alignment Kernels
    *   Distillation Kernels
    *   Experimental Kernels

---

## Examples

Check out the following examples to see Liger Kernel in action:

*   [Hugging Face Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)
*   [Lightning Trainer](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)
*   [Medusa Multi-head LLM (Retraining Phase)](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)
*   [Vision-Language Model SFT](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)
*   [Liger ORPO Trainer](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)

---

## Support & Community

*   **Issues:**  Report issues on [GitHub](https://github.com/linkedin/Liger-Kernel/issues).
*   **Discussion:**  Join the discussion on our [Discord channel](https://discord.com/channels/1189498204333543425/1275130785933951039).
*   **Collaboration:** For formal collaborations, contact Yanning Chen (yannchen@linkedin.com) and Zhipeng Wang (zhipwang@linkedin.com).

---

## Acknowledgements

Special thanks to our sponsors and collaborators: [Glows.ai](https://platform.glows.ai/), [AMD](https://www.amd.com/en.html), [Intel](https://www.intel.com/), [Modal](https://modal.com/), [EmbeddedLLM](https://embeddedllm.com/), [HuggingFace](https://huggingface.co/), [Lightning AI](https://lightning.ai/), [Axolotl](https://axolotl.ai/), and [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory).

Refer to the [original README](https://github.com/linkedin/Liger-Kernel) for full details.

---

## License

Liger Kernel is licensed under the [MIT License](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md).