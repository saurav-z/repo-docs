<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference</h1>
  <p>Experience cutting-edge Large Language Model (LLM) inference optimizations with KTransformers, a flexible and efficient framework.</p>

  <p>
    <strong><a href="#show-cases">🌟 Show Cases</a> | <a href="#key-features">🔑 Key Features</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a> | <a href="#faq"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">💻 View on GitHub</a> </strong>
  </p>
</div>

## 🎉 Introduction

KTransformers, also known as Quick Transformers, is a Python-centric framework designed to boost your Hugging Face Transformers experience.  It offers advanced kernel optimizations and smart placement/parallelism strategies, allowing you to run LLMs faster and more efficiently. KTransformers is designed for extensibility, letting you easily integrate optimized modules for significant performance gains. It supports a Transformers-compatible interface, OpenAI and Ollama-compliant RESTful APIs, and even a user-friendly ChatGPT-like web UI.

## 🔥 Updates

*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025**: Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
*   **Feb 15, 2025**: Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025**: Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Aug 28, 2024**: Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024**: Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
*   **Aug 14, 2024**: Support llamfile as linear backend.
*   **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024**: Support windows native.

## 🔑 Key Features

*   **Optimized Kernels:** Integrate advanced kernels to accelerate inference.
*   **Flexible Injection Framework:** Easily swap out modules for optimized versions with minimal code changes.
*   **Transformers Compatibility:** Use KTransformers with the familiar Hugging Face Transformers interface.
*   **RESTful APIs:** Compatible with OpenAI and Ollama APIs.
*   **User-Friendly UI:** Simplified ChatGPT-like web UI.
*   **Heterogeneous Computing Support:** Leverages GPU/CPU offloading for efficient use of resources.
*   **Support for Quantization:**  Efficiently run models using techniques like GGUF/GGML, Llamafile, and Marlin.

## 🌟 Show Cases

KTransformers delivers impressive performance improvements and enables running large models on local machines.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):** Up to 286.55 tokens/s with optimized AMX-based MoE kernel. Achieving up to **27.79x speedup** compared to llama.cpp.
    *   **Decode Speed (tokens/s):** Up to 13.69 tokens/s with selective expert activation. Achieving up to **3.03x speedup** compared to llama.cpp.
    *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4_K_M version using only 21GB VRAM and 136GB DRAM.

    *   Scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).
    *   Achieves 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading.
    *   Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via OpenAI and Ollama compatible API.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **GPT-4/o1-level Local VSCode Copilot:** Experience a powerful local coding assistant on a desktop with only 24GB VRAM.

<p align="center">
  <picture>
    <img alt="Copilot Demo" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=60%>
  </picture>
</p>

<!--
*   **1M Context Local Inference on a Desktop with Only 24GB VRAM**
    *   **1M Context InternLM 2.5 7B**: Operates at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM, which is feasible on a local desktop setup. It achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.
    *   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.
    *   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
-->

## 🚀 Quick Start

### 📥 Installation

Install KTransformers following the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Tutorial:  Injection Framework

KTransformers' core is a user-friendly, template-based injection framework, simplifying module replacement with optimized variants. This facilitates combining optimizations and exploring their synergistic effects.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments with limited resources, emphasizing heterogeneous computing (GPU/CPU offloading of quantized models), supporting efficient kernels like <a href="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a href="https://github.com/IST-DASLab/marlin">Marlin</a>.

### Example Usage

Use provided kernels by creating a YAML-based injection template and adding a call to `optimize_and_load_gguf`:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

`optimize_and_load_gguf` iterates through the model's sub-modules, matches YAML rules, and replaces them with advanced modules.  The original `generate` interface remains available, alongside `prefill_and_generate` for further optimizations (e.g., CUDAGraph).

### Customizing Your Model

Find detailed injection and multi-GPU tutorials (DeepSeek-V2 example) [here](doc/en/injection_tutorial.md).

Here's a YAML template example:

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

Each rule has `match` and `replace` parts; `match` specifies the module to replace, `replace` specifies the injected module and initialization keywords.  Example rule templates are in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For the design principles and implementation details, refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🤝 Acknowledgement and Contributors

KTransformers is built upon the foundation of Hugging Face Transformers and benefits from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  The project is actively maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  We welcome new contributions!

## 🙋 Discussion

For questions, open an issue or join our WeChat group (QR code below).

<p align="center">
  <img alt="WeChat Group QR Code" src="WeChatGroup.png" width=20%>
</p>

## 🙋 FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).