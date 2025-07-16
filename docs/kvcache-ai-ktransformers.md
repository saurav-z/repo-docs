<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h2>KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations</h2>
  <p><b>KTransformers</b> provides a flexible framework for accelerating Large Language Model (LLM) inference, enabling faster and more efficient performance on your hardware. <a href="https://github.com/kvcache-ai/ktransformers"><b>Explore KTransformers on GitHub!</b></a></p>
    <p>
    <a href="#show-cases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion </a> | <a href="#faq">🙋 FAQ</a>
  </p>
</div>

## Key Features

*   **Optimized Kernel Integration:** Easily integrate advanced kernel optimizations for significant speedups.
*   **Transformers Compatibility:** Seamlessly works with Hugging Face Transformers for a familiar user experience.
*   **Flexible & Extensible:** Designed for easy customization and experimentation with different optimization techniques.
*   **OpenAI & Ollama API Compliance:** Provides RESTful APIs compatible with popular LLM interfaces.
*   **Simplified Web UI:** Includes a user-friendly web UI for easy interaction (similar to ChatGPT).
*   **Hardware Acceleration:** Supports a variety of hardware, including Intel Arc GPU, AMD ROCm, and more.

## What's New

*   **July 11, 2025:** Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025:** Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025:** Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025:** Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025:** Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
*   **Feb 15, 2025:** Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025:** Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Aug 28, 2024:** Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024:** Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
*   **Aug 14, 2024:** Support llamfile as linear backend.
*   **Aug 12, 2024:** Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024:** Support windows native.

## Show Cases

KTransformers delivers impressive performance improvements for LLMs on various hardware configurations.

### GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

<p align="center">
  <picture>
    <img alt="GPT-4/o1-level Local VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=70%>
  </picture>
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run Q4_K_M version with only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   Prefill Speed (tokens/s):
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   Decode Speed (tokens/s):
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   Upcoming Open Source Release:
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run Q4\_K\_M version with only 21GB VRAM and 136GB DRAM, rivaling GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=70%>
  </picture>
</p>

*   **Faster Speed:** 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation, leveraging MoE offloading and advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via an OpenAI/Ollama-compatible API.

<p align="center">
  <picture>
    <img alt="VSCode Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=70%>
  </picture>
</p>

**More advanced features are coming soon!**

## Quick Start

Get up and running with KTransformers in a few simple steps.

### Installation

For detailed installation instructions, please refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

### Supported Vendors:
- Metax
- Sanechips (ZhuFeng V1.0)
- Intel
- Ascend
- Kunpeng
- AMD

## Tutorial: Injection Framework

KTransformers features a user-friendly, template-based injection framework for optimizing Transformers modules.

<p align="center">
  <picture>
    <img alt="Injection Framework Overview" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments, especially GPU/CPU offloading of quantized models. It supports efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels. More details [here](doc/en/operators/llamafile.md).

### Example Usage

Use YAML-based injection templates and the `optimize_and_load_gguf` function.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

The `optimize_and_load_gguf` function replaces modules based on rules in your YAML file.

### Customization

A detailed injection and multi-GPU tutorial using DeepSeek-V2 is available [here](doc/en/injection_tutorial.md).

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

The YAML rules define `match` (which module to replace) and `replace` (the optimized module).

Find example rule templates for DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For design principles, see the [design document](doc/en/deepseek-v2-injection.md).

## Acknowledgments & Contributors

KTransformers leverages the Transformers library and advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute back to the community.

KTransformers is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome new contributors!

## Discussion

Have questions? Open an issue or join our WeChat group! QR Code: [WeChat Group](WeChatGroup.png)

## FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).