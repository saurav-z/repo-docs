<div align="center">
  <p align="center">
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </p>
  <h1>KTransformers: Accelerate LLM Inference with Advanced Optimizations</h1>
  <p><strong>Unlock blazing-fast LLM performance with KTransformers, your gateway to cutting-edge inference optimizations!</strong></p>

  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#FAQ">🙋 FAQ</a>
    <br>
    <a href="https://github.com/kvcache-ai/ktransformers">🚀 View on GitHub</a>
  </p>
</div>

## 🎉 Introduction

KTransformers is a Python-centric framework designed to supercharge your Hugging Face 🤗 Transformers experience. It provides a flexible platform for experimenting with innovative Large Language Model (LLM) inference optimizations, including kernel optimizations, advanced placement/parallelism strategies, and support for various hardware backends. Enhance your workflows with a Transformers-compatible interface, and integrations for RESTful APIs (compatible with OpenAI and Ollama), and a simplified ChatGPT-like web UI.

**Key Features:**

*   **Optimized Kernels:** Leverage advanced kernels for significant performance gains.
*   **Flexible Injection:** Easily inject optimized modules with a single line of code.
*   **Transformers Compatibility:** Seamless integration with the Hugging Face Transformers ecosystem.
*   **OpenAI & Ollama API Compliance:** Supports standard API interfaces for easy integration.
*   **Simplified Web UI:** Provides a user-friendly interface for interacting with LLMs.
*   **Hardware Support:** Support for various vendors including Intel, AMD, and more.

## 🔥 Updates

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
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

## 🌟 Show Cases

KTransformers showcases exceptional performance improvements for LLMs on consumer hardware.

<div>
<h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run its Q4_K_M version with just 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   Prefill Speed (tokens/s):
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   Decode Speed (tokens/s):
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   Upcoming Open Source Release: AMX optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Run its Q4_K_M version with only 21GB VRAM and 136GB DRAM on a local desktop machine.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Wrapped into an OpenAI and Ollama compatible API for seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and various other frontends.

<p align="center">
  <!-- Add your desired image link here -->
</p>
<!-- Add your desired image link here -->

**Stay tuned for more advanced features!**

## 🚀 Quick Start

KTransformers offers a straightforward setup process.

### 📥 Installation

See the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for details.

## 📃 Brief Injection Tutorial

KTransformers simplifies LLM optimization through a template-based injection framework. It enables easy replacement of original torch modules with optimized variants, promoting a modular approach to experimentation.

</br>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on optimizing local deployments. The framework supports heterogeneous computing by offloading quantized models to GPU/CPU. For example, <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU are supported. Details are <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Integrate provided kernels by creating a YAML-based injection template and invoking `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

The AutoModel is initialized on the meta device to avoid memory usage.  `optimize_and_load_gguf` matches and replaces modules with advanced modules.

After injection, the `generate` interface is available.  A `prefill_and_generate` method enables further optimizations.

### How to Custom Your Model

A tutorial on injection and multi-GPU usage with DeepSeek-V2 is [here](doc/en/injection_tutorial.md).

Example YAML template to replace all Linear modules with Marlin:

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

Each rule uses `match` and `replace`.  The `match` specifies the module and `replace` the module with initialization keywords.

Example rule templates for DeepSeek-V2 and Qwen2-57B-A14 are in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For design principles and implementation details, see the [design document](doc/en/deepseek-v2-injection.md).

##  Acknowledgment and Contributors

KTransformers is built upon the Transformers framework and benefits from contributions by GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. We aim to contribute to the community.

KTransformers is maintained by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome contributors.

## 🙋 FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).