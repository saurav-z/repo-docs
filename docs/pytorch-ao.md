# TorchAO: Accelerate AI Models with PyTorch-Native Optimization

**TorchAO empowers you to optimize PyTorch models from training to serving, achieving significant speedups and memory savings.**  Get started with the original repo on [GitHub](https://github.com/pytorch/ao).

**Key Features:**

*   ⚡ **Float8 Training & Inference:** Train Llama-3.1-70B **1.5x faster** with float8 training.
*   🧠 **Quantization-Aware Training (QAT):** Recover **77% of quantized perplexity degradation** on Llama-3-2-3B with QAT.
*   🚀 **Weight-Only Quantization (WOQ):** Quantize Llama-3-8B to int4 for **1.89x faster** inference with **58% less memory**.
*   🧩 **Sparsity Support:** Implement 2:4 and block sparsity.
*   ⚙️ **Integration:** Built-in with Hugging Face Transformers, vLLM, and more.
*   🛠️ **Easy to Use:** Achieve performance gains with minimal code changes.

---

## 📣 Latest News

Stay updated with the latest TorchAO developments:

*   [Jun 25] Our [TorchAO paper](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) was accepted to CodeML @ ICML 2025!
*   [May 25] QAT is now integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning ([docs](https://docs.axolotl.ai/docs/qat.html))!
*   [Apr 25] Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) at 2k H100 GPU scale
*   [Apr 25] TorchAO is added as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) ([docs](https://docs.vllm.ai/en/latest/features/quantization/torchao.html))!
*   [Mar 25] Our [2:4 Sparsity paper](https://openreview.net/pdf?id=O5feVk7p6Y) was accepted to SLLM @ ICLR 2025!
*   [Jan 25] Our [integration with GemLite and SGLang](https://pytorch.org/blog/accelerating-llm-inference/) yielded 1.1-2x faster inference with int4 and float8 quantization across different batch sizes and tensor parallel sizes
*   [Jan 25] We added [1-8 bit ARM CPU kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for linear and embedding ops

<details>
  <summary>Older news</summary>

- [Nov 24] We achieved [1.43-1.51x faster pre-training](https://pytorch.org/blog/training-using-float8-fsdp2/) on Llama-3.1-70B and 405B using float8 training
- [Oct 24] TorchAO is added as a quantization backend to HF Transformers!
- [Sep 24] We officially launched TorchAO. Check out our blog [here](https://pytorch.org/blog/pytorch-native-architecture-optimization/)!
- [Jul 24] QAT [recovered up to 96% accuracy degradation](https://pytorch.org/blog/quantization-aware-training/) from quantization on Llama-3-8B
- [Jun 24] Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively
- [Jun 24] Block sparsity [achieved 1.46x training speeedup](https://pytorch.org/blog/speeding-up-vits/) on the ViT model with <2% drop in accuracy

</details>

---

## 🌅 Overview

TorchAO is a PyTorch-native framework designed to optimize your AI models across the entire lifecycle, from training to serving. It leverages quantization and sparsity techniques for significant performance improvements. TorchAO seamlessly integrates with `torch.compile()` and `FSDP2` and is compatible with a wide range of Hugging Face PyTorch models.

**Key Capabilities:**

*   **Float8 Training and Inference:** Achieve substantial speedups without compromising accuracy.
*   **MX Training and Inference:**  Offers MX tensor formats based on native PyTorch MX dtypes (prototype).
*   **Quantization-Aware Training (QAT):** Mitigate accuracy degradation caused by quantization.
*   **Post-Training Quantization (PTQ):** Supports int4, int8, fp6, etc., with optimized kernels for various backends, including CUDA, ARM CPU, and XNNPACK.
*   **Sparsity:** Includes techniques like 2:4 and block sparsity for further optimization.

Explore detailed information in our [docs](https://docs.pytorch.org/ao/main/).

TorchAO is brought to you by the team behind the "fast" series:

*   sam-fast
*   gpt-fast
*   sd-fast
*   flux-fast
*   seamlessv2-fast

---

## 🚀 Quick Start

Get up and running with TorchAO in minutes:

1.  **Install TorchAO:**

    ```bash
    pip install torchao
    ```

2.  **Quantize your model to int4 (example):**

    ```python
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    quantize_(model, Int4WeightOnlyConfig(group_size=32))
    ```

    Expect substantial speed and memory gains, for example:

    ```
    int4 model size: 1.25 MB
    bfloat16 model size: 4.00 MB
    compression ratio: 3.2

    bf16 mean time: 30.393 ms
    int4 mean time: 4.410 ms
    speedup: 6.9x
    ```

    Refer to our [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html) for detailed setup and benchmarking information.  You can also experiment with your own model on our [Hugging Face space](https://huggingface.co/spaces/pytorch/torchao-my-repo)!

---

## 🛠 Installation

Install the latest stable version of TorchAO:

```bash
pip install torchao
```

<details>
  <summary>Other installation options</summary>

  ```bash
  # Nightly
  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126
  
  # Different CUDA versions
  pip install torchao --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
  pip install torchao --index-url https://download.pytorch.org/whl/cpu    # CPU only

  # For developers
  USE_CUDA=1 python setup.py develop
  USE_CPP=0 python setup.py develop
  ```
</details>

---

## 🔗 Integrations

TorchAO seamlessly integrates with leading open-source libraries:

*   Hugging Face Transformers (with a built-in inference backend and low-bit optimizers)
*   Hugging Face Diffusers (best practices with `torch.compile` and TorchAO)
*   Hugging Face PEFT (for LoRA using TorchAO as a quantization backend)
*   Mobius HQQ backend leveraged our int4 kernels to get [195 tok/s on a 4090](https://github.com/mobiusml/hqq#faster-inference)
*   TorchTune (for NF4 QLoRA, QAT, and float8 quantized fine-tuning recipes)
*   TorchTitan (for float8 pre-training)
*   VLLM (for LLM serving)
*   SGLang (for LLM serving)
*   Axolotl (for QAT and PTQ)

---

## 🔎 Inference

Boost your model's inference performance with TorchAO:

*   **Int4 weight-only:** Up to **1.89x throughput with 58.1% less memory** on Llama-3-8B.
*   **Float8 dynamic quantization:** Up to **1.54x and 1.27x speedup** on Flux.1-Dev\* and CogVideoX-5b respectively on H100.
*   **Int4 + 2:4 Sparsity:** Achieve **2.37x throughput with 67.7% memory reduction** on Llama-3-8B.

Quantize models with just one line of code (Option 1) or load pre-quantized models via Hugging Face (Option 2):

#### Option 1: Direct TorchAO API

```python
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig(group_size=128, use_hqq=True))
```

#### Option 2: HuggingFace Integration

```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization.quant_api import Int4WeightOnlyConfig

# Create quantization configuration
quantization_config = TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=128, use_hqq=True))

# Load and automatically quantize
quantized_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

#### Deploy quantized models in vLLM with one command:

```bash
vllm serve pytorch/Phi-4-mini-instruct-int4wo-hqq --tokenizer microsoft/Phi-4-mini-instruct -O3
```

Our quantization flow achieves a **67% VRAM reduction and 12-20% speedup** on A100 GPUs while maintaining model quality. Explore the [step-by-step quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe) and discover pre-quantized models [here](https://huggingface.co/pytorch).

---

## 🚅 Training

TorchAO provides powerful tools to accelerate and improve your training process:

### Quantization-Aware Training

QAT is critical to improve the accuracy of quantized models, especially for lower bit-width dtypes like int4. Our QAT recipe recovers a significant amount of accuracy lost in PTQ, in collaboration with [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat). Check the [QAT README](torchao/quantization/qat/README.md) and [original blog](https://pytorch.org/blog/quantization-aware-training/) for more information.

```python
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
from torchao.quantization.qat import QATConfig

# prepare
base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
quantize_(my_model, QATConfig(base_config, step="prepare"))

# train model (not shown)

# convert
quantize_(my_model, QATConfig(base_config, step="convert"))
```

Combine LoRA and QAT to speed up training by up to **1.89x** using the [fine-tuning recipe](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py).

### Float8

[torchao.float8](torchao/float8) provides training recipes with scaled float8 dtypes. Experience speedups of up to **1.5x** on up to 512 GPU / 405B parameter count scale with ``torch.compile`` ([details](https://pytorch.org/blog/training-using-float8-fsdp2/)):

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

Integrate our float8 training with [TorchTitan's pre-training flows](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md). Check out our blog posts about float8 training support:

*   [Accelerating Large Scale Training and Convergence with PyTorch Float8 Rowwise on Crusoe 2K H200s](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
*   [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
*   [Efficient Pre-training of Llama 3-like model architectures using torchtitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
*   [Float8 in PyTorch](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)

### Sparse Training

TorchAO enables semi-structured 2:4 sparsity, providing up to **6% end-to-end speedups on ViT-L**. Read more about our [accelerating neural network training here](https://pytorch.org/blog/accelerating-neural-network-training/).  Use one line of code with full examples available in [torchao/sparsity/training/](torchao/sparsity/training/):

```python
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-efficient optimizers

TorchAO tackles memory bottlenecks with two innovative approaches:

**1. Quantized optimizers:** Reduce optimizer state memory by 2-4x by quantizing to lower precision

```python
from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
```
See our detailed [benchmarks here](https://github.com/pytorch/ao/tree/main/torchao/optim).

**2. CPU offloading:** Move optimizer state and gradients to CPU memory to achieve significant memory savings.
```python
optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
optim.load_state_dict(ckpt["optim"])
```
Our single GPU CPU offloading can reduce your VRAM requirements by 60%.

---

## 🎥 Videos

Explore TorchAO with these informative videos:

*   [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
*   [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
*   [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
*   [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
*   [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
*   [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)

---

## 💬 Citation

If you leverage the TorchAO library, please cite it:

```bibtex
@software{torchao,
  title={TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
  author={torchao},
  url={https://github.com/pytorch/ao},
  license={BSD-3-Clause},
  month={oct},
  year={2024}
}