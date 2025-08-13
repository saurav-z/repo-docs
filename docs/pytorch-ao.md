<div align="center">

# TorchAO: Optimize PyTorch Models for Speed and Efficiency

</div>

Accelerate your PyTorch models from training to serving with TorchAO, a PyTorch-native framework offering quantization, sparsity, and more. [Explore the TorchAO repository](https://github.com/pytorch/ao).

*   **Pre-train Llama-3-70B 1.5x faster** with float8 training.
*   **Recover 77% of quantized perplexity degradation** on Llama-3-3B using QAT.
*   **Quantize Llama-3-8B to int4 for 1.89x faster inference** and **58% less memory usage.**

<div align="center">

[![CodeML @ ICML 2025](https://img.shields.io/badge/CodeML_%40_ICML-2025-blue)](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf)
[![TorchAO in GPU Mode](https://dcbadge.vercel.app/api/server/gpumode?style=flat&label=TorchAO%20in%20GPU%20Mode)](https://discord.com/channels/1189498204333543425/1205223658021458100)
[![Contributors](https://img.shields.io/github/contributors-anon/pytorch/ao?color=yellow&style=flat-square)](https://github.com/pytorch/ao/graphs/contributors)
[![Documentation](https://img.shields.io/badge/torchao-documentation-blue?color=DE3412)](https://docs.pytorch.org/ao/stable/index.html)
[![License](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)

</div>

## Key Features

*   **Float8 Training and Inference:** Achieve significant speedups without sacrificing accuracy. ([torchao/float8/README.md](torchao/float8/README.md), [Float8 docs](https://docs.pytorch.org/ao/main/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html))
*   **MX Training and Inference:** Utilize MX tensor formats based on native PyTorch MX dtypes (prototype). ([torchao/prototype/mx_formats/README.md](torchao/prototype/mx_formats/README.md))
*   **Quantization-Aware Training (QAT):** Mitigate quantization degradation for improved accuracy. ([torchao/quantization/qat/README.md](torchao/quantization/qat/README.md))
*   **Post-Training Quantization (PTQ):** Quantize to int4, int8, fp6, and more, with optimized kernels for CUDA, ARM CPU, and XNNPACK. ([torchao/quantization/README.md](torchao/quantization/README.md))
*   **Sparsity:** Explore various techniques, including 2:4 and block sparsity. ([torchao/sparsity/README.md](torchao/sparsity/README.md))

## 📣 Latest News

*   **[June 2024]** Our [TorchAO paper](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) was accepted to CodeML @ ICML 2025!
*   **[May 2024]** QAT is now integrated into [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning ([docs](https://docs.axolotl.ai/docs/qat.html))!
*   **[April 2024]** Float8 rowwise training yielded [1.34-1.43x training speedup](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) at 2k H100 GPU scale
*   **[April 2024]** TorchAO is added as a [quantization backend to vLLM](https://docs.vllm.ai/en/latest/features/quantization/torchao.html) ([docs](https://docs.vllm.ai/en/latest/features/quantization/torchao.html))!
*   **[March 2024]** Our [2:4 Sparsity paper](https://openreview.net/pdf?id=O5feVk7p6Y) was accepted to SLLM @ ICLR 2025!
*   **[January 2024]** Our [integration with GemLite and SGLang](https://pytorch.org/blog/accelerating-llm-inference/) yielded 1.1-2x faster inference with int4 and float8 quantization across different batch sizes and tensor parallel sizes
*   **[January 2024]** We added [1-8 bit ARM CPU kernels](https://pytorch.org/blog/hi-po-low-bit-operators/) for linear and embedding ops

<details>
  <summary>Older news</summary>

*   **[Nov 2023]** We achieved [1.43-1.51x faster pre-training](https://pytorch.org/blog/training-using-float8-fsdp2/) on Llama-3.1-70B and 405B using float8 training
*   **[Oct 2023]** TorchAO is added as a quantization backend to HF Transformers!
*   **[Sep 2023]** We officially launched TorchAO. Check out our blog [here](https://pytorch.org/blog/pytorch-native-architecture-optimization/)!
*   **[Jul 2023]** QAT [recovered up to 96% accuracy degradation](https://pytorch.org/blog/quantization-aware-training/) from quantization on Llama-3-8B
*   **[Jun 2023]** Semi-structured 2:4 sparsity [achieved 1.1x inference speedup and 1.3x training speedup](https://pytorch.org/blog/accelerating-neural-network-training/) on the SAM and ViT models respectively
*   **[Jun 2023]** Block sparsity [achieved 1.46x training speeedup](https://pytorch.org/blog/speeding-up-vits/) on the ViT model with <2% drop in accuracy

</details>

## 🚀 Quick Start

Get started with TorchAO in two simple steps:

1.  **Install:**
    ```bash
    pip install torchao
    ```
2.  **Quantize your model:**
    ```python
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    quantize_(model, Int4WeightOnlyConfig(group_size=32))
    ```

    Achieve significant speedups and compression.  For example, a quantized model can be 6.9x faster and 3.2x smaller. Check out the [quick start guide](https://docs.pytorch.org/ao/stable/quick_start.html).  Also, try quantizing models using our [HuggingFace space](https://huggingface.co/spaces/pytorch/torchao-my-repo)!

## 🛠 Installation

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

## 🔗 Integrations

TorchAO seamlessly integrates with popular libraries:

*   HuggingFace Transformers ([builtin inference backend](https://huggingface.co/docs/transformers/main/quantization/torchao), [low bit optimizers](https://github.com/huggingface/transformers/pull/31865))
*   HuggingFace Diffusers ([diffusers-torchao](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/torchao.md))
*   HuggingFace PEFT ([quantization backend](https://huggingface.co/docs/peft/en/developer_guides/quantization#torchao-pytorch-architecture-optimization))
*   Mobius HQQ ([195 tok/s on a 4090](https://github.com/mobiusml/hqq#faster-inference))
*   TorchTune (QLoRA, QAT, and float8 quantized fine-tuning)
*   TorchTitan (float8 pre-training)
*   VLLM ([usage](https://docs.vllm.ai/en/latest/features/quantization/torchao.html), [detailed docs](https://docs.pytorch.org/ao/main/torchao_vllm_integration.html))
*   SGLang ([usage](https://docs.sglang.ai/backend/server_arguments.html#server-arguments) and [PR](https://github.com/sgl-project/sglang/pull/1341))
*   Axolotl ([QAT](https://docs.axolotl.ai/docs/qat.html) and [PTQ](https://docs.axolotl.ai/docs/quantize.html))

## 🔎 Inference

TorchAO provides substantial performance gains with minimal changes to your code:

*   **Int4 weight-only:** [1.89x throughput with 58.1% less memory](torchao/quantization/README.md) on Llama-3-8B
*   **Float8 dynamic quantization:** [1.54x and 1.27x speedup on Flux.1-Dev* and CogVideoX-5b respectively](https://github.com/sayakpaul/diffusers-torchao) on H100 with preserved quality
*   **Int4 + 2:4 Sparsity:** [2.37x throughput with 67.7% memory reduction](torchao/sparsity/README.md) on Llama-3-8B

Quantize your model with one line (Option 1) or load a quantized model directly from Hugging Face (Option 2):

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

```shell
vllm serve pytorch/Phi-4-mini-instruct-int4wo-hqq --tokenizer microsoft/Phi-4-mini-instruct -O3
```

Achieve **67% VRAM reduction and 12-20% speedup** on A100 GPUs while maintaining model quality. See [this quantization guide](https://huggingface.co/pytorch/Phi-4-mini-instruct-int4wo-hqq#quantization-recipe) and pre-quantized models [here](https://huggingface.co/pytorch).

## 🚅 Training

### Quantization-Aware Training (QAT)

QAT improves accuracy for quantized models.  Our QAT recipe, developed with [TorchTune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md#quantization-aware-training-qat), recovers **96% of accuracy degradation on hellaswag and 68% of the perplexity degradation on wikitext** for Llama3 compared to post-training quantization (PTQ).

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

Combine LoRA + QAT to speed up training by [1.89x](https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700) using this [fine-tuning recipe](https://github.com/pytorch/torchtune/blob/main/recipes/qat_lora_finetune_distributed.py).

### Float8 Training

Leverage float8 dtypes for training speedups. Our float8 training is integrated into [TorchTitan's pre-training flows](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md).

```python
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(m)
```

See:
*   [Accelerating Large Scale Training and Convergence with PyTorch Float8 Rowwise on Crusoe 2K H200s](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)
*   [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
*   [Efficient Pre-training of Llama 3-like model architectures using torchtitan on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)
*   [Float8 in PyTorch](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)

### Sparse Training

Implement semi-structured 2:4 sparsity for **6% end-to-end speedups on ViT-L**.

```python
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
swap_linear_with_semi_sparse_linear(model, {"seq.0": SemiSparseLinear})
```

### Memory-Efficient Optimizers

Reduce GPU memory usage:

1.  **Quantized optimizers:** Reduce optimizer state memory by 2-4x.

    ```python
    from torchao.optim import AdamW8bit, AdamW4bit, AdamWFp8
    optim = AdamW8bit(model.parameters()) # replace with Adam4bit and AdamFp8 for the 4 / fp8 versions
    ```

    See [benchmarks here](https://github.com/pytorch/ao/tree/main/torchao/optim).

2.  **CPU offloading:** Move optimizer state and gradients to CPU memory.

    ```python
    optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, fused=True)
    optim.load_state_dict(ckpt["optim"])
    ```

    This can **reduce your VRAM requirements by 60%**.

<!--
## For Developers
... (removed developer section)
-->
## 🎥 Videos

*   [Keynote talk at GPU MODE IRL](https://youtu.be/FH5wiwOyPX4?si=VZK22hHz25GRzBG1&t=1009)
*   [Low precision dtypes at PyTorch conference](https://youtu.be/xcKwEZ77Cps?si=7BS6cXMGgYtFlnrA)
*   [Slaying OOMs at the Mastering LLM's course](https://www.youtube.com/watch?v=UvRl4ansfCg)
*   [Advanced Quantization at CUDA MODE](https://youtu.be/1u9xUK3G4VM?si=4JcPlw2w8chPXW8J)
*   [Chip Huyen's GPU Optimization Workshop](https://www.youtube.com/live/v_q2JTIqE20?si=mf7HeZ63rS-uYpS6)
*   [Cohere for AI community talk](https://www.youtube.com/watch?v=lVgrE36ZUw0)

## 💬 Citation

```bibtex
@software{torchao,
  title={TorchAO: PyTorch-Native Training-to-Serving Model Optimization},
  author={torchao},
  url={https://github.com/pytorch/ao},
  license={BSD-3-Clause},
  month={oct},
  year={2024}
}
```