<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Kernel Optimizations</h1>
  <p><em>Unlock faster and more efficient Large Language Model (LLM) inference with KTransformers, a flexible framework for cutting-edge optimization techniques.</em></p>

  <p>
    <strong><a href="#show-cases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion </a> | <a href="#FAQ">🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">🔗 View on GitHub</a></strong>
  </p>
</div>

## Key Features

*   🚀 **Blazing-Fast Inference:** Achieve significant speedups for LLM inference through advanced kernel optimizations.
*   🛠️ **Flexible Framework:** Easily integrate optimized modules with a single line of code, leveraging a Transformers-compatible interface.
*   🔄 **OpenAI & Ollama Compatibility:** Supports RESTful APIs compliant with OpenAI and Ollama for seamless integration.
*   🌐 **Simplified Web UI:** Includes a streamlined, ChatGPT-like web UI for easy interaction with optimized models.
*   🧠 **Heterogeneous Computing:** Optimized for GPU/CPU offloading of quantized models, enhancing performance on various hardware.
*   🧩 **Extensible Design:** Built with extensibility at its core, allowing for easy experimentation with new LLM inference optimizations.

## 🔥 Recent Updates

*   **July 26, 2025:** Support for SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support for Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) prefix cache reuse.
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support AMX-Int8, AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025:** Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025:** Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025:** Support unsloth 1.58/2.51 bits weights and IQ1_S/FP8 hybrid weights. Support 139K Longer Context for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025:** Support FP8 GPU kernel for DeepSeek-V3 and R1; Longer Context.
*   **Feb 15, 2025:** Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed, update docs and online books.
*   **Feb 10, 2025:** Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup.
*   **Aug 28, 2024:** Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024:** Update detailed tutorial for injection and multi-GPU.
*   **Aug 14, 2024:** Support llamfile as linear backend.
*   **Aug 12, 2024:** Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024:** Support windows native.

## 🌟 Show Cases

KTransformers enables impressive performance improvements for local LLM deployments.

### Local LLM Inference:

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4_K_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    -   **Prefill Speed (tokens/s):**
        -   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        -   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    -   **Decode Speed (tokens/s):**
        -   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        -   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    -   **Upcoming Open Source Release:**
        -   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        -   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).

*   **Local 236B DeepSeek-Coder-V2:** Running its Q4_K_M version using only 21GB VRAM and 136GB DRAM, performing even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
    </p>
    -   **Faster Speed:** Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    -   **VSCode Integration:** Seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c"  />
    </p>

###

**More advanced features are coming soon, stay tuned!**

## 🚀 Quick Start

Get up and running with KTransformers in no time!

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Brief Injection Tutorial

KTransformers leverages a template-based injection framework for easy module optimization.

</br>
<p align="center">
  <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</p>

KTransformers focuses on local deployments with limited resources, emphasizing heterogeneous computing opportunities.  It supports efficient kernels like <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a>.

### Example Usage

Use YAML templates and `optimize_and_load_gguf` to integrate optimized kernels.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

Customize your model using a YAML template, as shown below:

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

Find example templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. For design principles, refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🙋 Acknowledgement and Contributors

KTransformers is built upon the foundation of the 🤗 Transformers library and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. The project is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  We welcome new contributors!

## 💬 Discussion

If you have any questions, please open an issue. You can also join our WeChat group for further discussion.  QR Code: [WeChat Group](WeChatGroup.png)

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).