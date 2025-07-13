<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference</h1>
  <p><strong>Optimize and accelerate your Hugging Face Transformers experience with advanced kernel optimizations and flexible deployment strategies.</strong></p>
  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">🔗 Original Repo</a>
  </p>
</div>

## 🎉 Introduction

KTransformers, or Quick Transformers, is a Python-centric framework designed to significantly enhance your experience with Hugging Face Transformers.  It provides a flexible platform for experimenting with cutting-edge LLM inference optimizations, offering a user-friendly interface for injecting advanced kernels and strategies. Easily integrate optimized modules with a single line of code to unlock a Transformers-compatible interface, RESTful APIs, and even a simplified ChatGPT-like web UI.

**Key Features:**

*   **Optimized Kernel Injection:** Easily integrate optimized modules into your Transformers models.
*   **Transformers Compatibility:** Works seamlessly with Hugging Face Transformers models.
*   **OpenAI and Ollama API Compliance:** Compatible with popular API standards.
*   **Flexible and Extensible:** Designed with extensibility at its core for experimentation.
*   **Web UI Support:** Includes a simplified ChatGPT-like web UI.
*   **Supports a wide range of vendors:** Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, AMD

## 🔥 Recent Updates

*   **July 11, 2025:** Support Kimi-K2 ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) prefix cache reuse ([Tutorial](./doc/en/prefix_cache.md))
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md))
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md))
*   **Apr 2, 2025:** Support Multi-concurrency ([Tutorial](./doc/en/balance-serve.md))
*   **Mar 15, 2025:** Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md))
*   **Mar 5, 2025:** Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025:** Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
*   **Feb 15, 2025:** Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025:** Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Aug 28, 2024:** Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024:** Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
*   **Aug 14, 2024:** Support llamfile as linear backend.
*   **Aug 12, 2024:** Support multiple GPU; Support new model: mixtral 8\*7B and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024:** Support windows native.

## 🌟 Show Cases

**Experience cutting-edge LLM performance on your local machine!**

### GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

<p align="center">
  <picture>
    <img alt="VSCode Copilot Demo" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=75%>
  </picture>
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Performance Boosts:**
        *   Prefill Speed (tokens/s): Up to 286.55 with AMX-based MoE kernel optimizations.
        *   Decode Speed (tokens/s): Up to 13.69 tokens/s with selective expert activation.
        *   Achieve up to **27.79x speedup** over llama.cpp in prefill and up to **3.03x speedup** during decoding.
    *   **Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3 (available now as a preview binary distribution).

*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version with only 21GB VRAM and 136GB DRAM.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

    *   **BigCodeBench Performance:** Outperforms GPT-4-0613.
    *   **High Speed:** Achieves 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    *   **Seamless Integration:** Integrated as a backend for Tabby and other frontends via OpenAI and Ollama compatible API.

<p align="center">
  <picture>
    <img alt="KTransformers Demo" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=75%>
  </picture>
</p>

**More advanced features will be coming soon!**

## 🚀 Quick Start

Get started with KTransformers in a few simple steps!

### 📥 Installation

Install KTransformers by following the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Brief Injection Tutorial

KTransformers simplifies LLM optimization through its user-friendly, template-based injection framework.  This allows researchers to easily replace original torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

We focus on local deployments with limited resources. We offer GPU/CPU offloading of quantized models such as Llamafile and Marlin kernels.

### Example Usage

Use `optimize_and_load_gguf` and create a YAML-based injection template.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

After injection, the original `generate` interface is available, but we also provide a compatible `prefill_and_generate` method, which enables further optimizations like CUDAGraph to improve generation speed.

### How to Customize Your Model

Find a detailed tutorial on injection and multi-GPU use [here](doc/en/injection_tutorial.md).

Example YAML template for replacing Linear modules with Marlin:

```yaml
- match:
    name: "^model\\.layers\\..*$"  # regular expression
    class: torch.nn.Linear  # only match modules matching name and class simultaneously
  replace:
    class: ktransformers.operators.linear.KTransformerLinear  # optimized Kernel on quantized data types
    device: "cpu"   # which devices to load this module when initializing
    kwargs:
      generate_device: "cuda"
      generate_linear_type: "QuantizedLinearMarlin"
```

Rules have `match` and `replace` parts.  See example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For design principles, refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🙏 Acknowledgment and Contributors

KTransformers leverages the Transformers framework, along with kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute our modifications back to the community.

KTransformers is actively maintained and developed by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/).

## 💬 Discussion

If you have any questions, open an issue or join our WeChat group (QR Code: [WeChat Group](WeChatGroup.png)) for discussion.

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).