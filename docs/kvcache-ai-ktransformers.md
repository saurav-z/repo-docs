<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h1>KTransformers: Supercharge Your LLM Inference with Optimized Kernels</h1>
  <p><em>Experience cutting-edge LLM inference optimization with KTransformers, a flexible and extensible framework.</em></p>

  <p>
    <strong><a href="#show-cases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a>|<a href="#FAQ"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">🔗  View on GitHub</a></strong>
  </p>
</div>

## 🎉 Introduction

KTransformers, pronounced "Quick Transformers," is your gateway to significantly faster and more efficient Large Language Model (LLM) inference. Built on a flexible, Python-centric framework, KTransformers empowers you to enhance your 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers) experience with advanced kernel optimizations, sophisticated placement strategies, and powerful parallelism techniques.  Easily integrate optimized modules with a single line of code and unlock a Transformers-compatible interface, OpenAI and Ollama RESTful APIs, and a user-friendly web UI.

**Key Features:**

*   **Optimized Kernels:** Leverage advanced kernels for significant speedups in LLM inference.
*   **Extensible Architecture:**  A flexible Python-centric framework designed for easy customization and integration of new optimizations.
*   **Transformers Compatibility:**  Seamlessly integrates with Hugging Face Transformers.
*   **API Support:** Includes RESTful APIs compatible with OpenAI and Ollama.
*   **Simplified UI:** Offers a ChatGPT-like web UI for easy interaction.
*   **Heterogeneous Computing Support:** Efficiently utilizes GPU/CPU resources for quantized models.

## 🔥 Recent Updates

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025:** Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025:** Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025:** Support unsloth 1.58/2.51 bits weights and [IQ1\_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025:** Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
*   **Feb 15, 2025:** Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025:** Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Aug 28, 2024:** Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024:** Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
*   **Aug 14, 2024:** Support llamfile as linear backend.
*   **Aug 12, 2024:** Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024:** Support windows native.

## 🌟 Show Cases

**[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM. ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md))

*   **Prefill Speed (tokens/s):**
    *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
    *   Up to **27.79× speedup** compared to llama.cpp (10.31 tokens/s on 2×32 cores).
*   **Decode Speed (tokens/s):**
    *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
    *   Up to **3.03× speedup** compared to llama.cpp (4.51 tokens/s on 2×32 cores).
*   **Upcoming Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3. Currently available in preview binary distribution.

**Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version with only 21GB VRAM and 136GB DRAM, achieving better scores than GPT-4-0613 on [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends through an OpenAI and Ollama compatible API.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" alt="VSCode Copilot Demo" width="60%">
</p>

**More advanced features are on the way! Stay tuned.**

## 🚀 Quick Start

Get started with KTransformers quickly:

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

## 📃 Brief Injection Tutorial

KTransformers features a template-based injection framework, making it easy to swap original torch modules with optimized variants and combine optimizations.

<p align="center">
  <picture>
    <img alt="Injection Architecture" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments and heterogeneous computing opportunities like GPU/CPU offloading of quantized models. We support efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

Use YAML-based injection templates with `optimize_and_load_gguf`:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

`optimize_and_load_gguf` replaces modules based on your YAML rules. After injection, use the original `generate` or the optimized `prefill_and_generate` methods.

### Customizing Your Model

See the detailed tutorial for injection using DeepSeek-V2 [here](doc/en/injection_tutorial.md).

Example YAML template for replacing Linear modules with Marlin:

```yaml
- match:
    name: "^model\\.layers\\..*$"  # regular expression
    class: torch.nn.Linear  # match modules matching name and class simultaneously
  replace:
    class: ktransformers.operators.linear.KTransformerLinear  # optimized Kernel on quantized data types
    device: "cpu"   # which devices to load this module when initializing
    kwargs:
      generate_device: "cuda"
      generate_linear_type: "QuantizedLinearMarlin"
```

Each rule uses `match` and `replace`.  The `match` part defines the modules to replace, and the `replace` part defines the replacement modules.  Find example rule templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.  Refer to the [design document](doc/en/deepseek-v2-injection.md) for design principles.

## 🤝 Acknowledgment and Contributors

KTransformers builds on the Transformers framework and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  We plan to contribute back to the community.

KTransformers is developed by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and [Approaching.AI](http://approaching.ai/). We welcome new contributors.

## 💬 Discussion

For questions, open an issue.  You can also join our WeChat group (QR code below) for discussions.

<img src="WeChatGroup.png" alt="WeChat Group QR Code" width="100">

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).