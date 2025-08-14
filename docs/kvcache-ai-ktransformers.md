<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference</h1>
  <p><strong>Experience cutting-edge LLM inference optimizations with KTransformers, a flexible and efficient framework.</strong></p>
  <p>
      <a href="#key-features">🔑 Key Features</a> |
      <a href="#showcases">🌟 Show Cases</a> |
      <a href="#quick-start">🚀 Quick Start</a> |
      <a href="#tutorial">📃 Tutorial</a> |
      <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion </a> |
      <a href="#faq">🙋 FAQ</a> |
      <a href="https://github.com/kvcache-ai/ktransformers">GitHub Repository</a>
  </p>
</div>

## Overview

KTransformers, pronounced "Quick Transformers," is a Python-centric framework designed to accelerate Large Language Model (LLM) inference. It offers a flexible platform for experimenting with and implementing advanced kernel optimizations, placement, and parallelism strategies, all within a user-friendly interface compatible with Hugging Face Transformers, OpenAI/Ollama APIs, and a simplified web UI. [Visit the KTransformers GitHub Repository](https://github.com/kvcache-ai/ktransformers) for more details.

## <a id="key-features"></a>🔑 Key Features

*   **Flexible Architecture:** Designed for easy extensibility and integration of optimized modules with a single line of code.
*   **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   **API Compliance:** Supports RESTful APIs compliant with OpenAI and Ollama.
*   **Simplified UI:** Provides a streamlined, ChatGPT-like web UI for easy interaction.
*   **Heterogeneous Computing Support:** Optimizes for GPU/CPU offloading of quantized models, including Llamafile and Marlin kernels.
*   **Advanced Kernel Integration:** Leverages cutting-edge kernels for significant performance gains.
*   **Multi-GPU Support:** Enables efficient model execution across multiple GPUs.
*   **Long Context Support:** Optimized for handling extended context lengths.
*   **Modular Injection:** Allows you to inject optimized modules with YAML-based configuration.

## Updates

*   **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))

https://github.com/user-attachments/assets/fafe8aec-4e22-49a8-8553-59fb5c6b00a2

*   **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).

https://github.com/user-attachments/assets/faa3bda2-928b-45a7-b44f-21e12ec84b8a

*   **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025**: Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
*   **Feb 15, 2025**: Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025**: Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Aug 28, 2024**: Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024**: Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
*   **Aug 14, 2024**: Support llamfile as linear backend.
*   **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024**: Support windows native.

## <a id="showcases"></a>🌟 Show Cases

KTransformers enables high-performance LLM inference on resource-constrained hardware.

### GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

<p align="center">
  <picture>
    <img alt="Copilot Demo" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=80%>
  </picture>
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   Prefill Speed (tokens/s):
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   Decode Speed (tokens/s):
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   Upcoming Open Source Release:
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Running its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, attainable on a local desktop machine, which scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Wrapped into an OpenAI and Ollama compatible API for seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and various other frontends.

<p align="center">
  <picture>
    <img alt="VSCode Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=80%>
  </picture>
</p>

## <a id="quick-start"></a>🚀 Quick Start

### 📥 Installation

Follow the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to set up KTransformers.

## <a id="tutorial"></a>📃 Injection Tutorial

KTransformers utilizes a template-based injection framework, allowing for easy integration of optimized modules.

<p align="center">
  <picture>
    <img alt="Injection Process" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

### Example Usage

To use the provided kernels, create a YAML-based injection template and call `optimize_and_load_gguf` before using the Transformers model:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

This example initializes the `AutoModel` on the meta device and then uses `optimize_and_load_gguf` to replace modules with optimized variants.  The original `generate` method is still available, and a `prefill_and_generate` method is provided for additional speed optimizations.

### How to customize your model

A detailed tutorial using DeepSeek-V2 as an example is available [here](doc/en/injection_tutorial.md).

Example YAML template:

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

Each rule has a `match` and `replace` section, allowing for targeted module replacement.  Example rule templates can be found in the [`ktransformers/optimize/optimize_rules`](ktransformers/optimize/optimize_rules) directory.

For more details, please refer to the [design document](doc/en/deepseek-v2-injection.md).

## <a id="ack"></a>Acknowledgment and Contributors

KTransformers builds upon the foundations of the Hugging Face Transformers and benefits from contributions from GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  The project is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  Contributions are welcome!

## <a id="discussion"></a>Discussion

Join the discussion and stay updated: Open an issue or join our WeChat group (QR Code: [WeChatGroup.png]).

## <a id="faq"></a>🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).