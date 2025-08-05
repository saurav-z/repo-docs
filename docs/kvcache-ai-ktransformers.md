<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

## KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations

**KTransformers is a flexible framework designed to accelerate your Hugging Face Transformers experience, offering advanced kernel optimizations for faster and more efficient LLM inference.**  [Explore the KTransformers Repository](https://github.com/kvcache-ai/ktransformers)

**[Show Cases](#show-cases) | [Quick Start](#quick-start) | [Tutorial](#tutorial) | [Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [FAQ](#FAQ)**

## Key Features

*   🚀 **Accelerated Inference:** Experience significant speedups through advanced kernel optimizations, including support for FP8, AMX, and more.
*   💡 **Flexible and Extensible:**  Easily integrate and experiment with cutting-edge LLM inference optimizations.  A Python-centric design promotes extensibility.
*   💻 **Transformers Compatibility:** Seamlessly integrate with the Hugging Face Transformers ecosystem.
*   🌐 **API Compliance:** Supports RESTful APIs compatible with OpenAI and Ollama, simplifying integration with existing tools.
*   🖼️ **Simplified UI:** Includes a simplified, ChatGPT-like web UI for easy interaction.
*   🔄 **Heterogeneous Computing:** Optimize models on GPU/CPU with features like GPU/CPU offloading and support for Llamafile and Marlin kernels for CPU and GPU, respectively.
*   🛠️ **Customization:**  Easily inject optimized modules with a single line of code using a template-based injection framework.

## Updates

*   **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
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
*   **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024**: Support windows native.

## 🌟 Show Cases

**Experience state-of-the-art LLM performance on your local machine!**

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version with only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4_K_M version using only 21GB VRAM and 136GB DRAM, comparable to GPT-4-0613 performance (achieved on a local desktop).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Integrate seamlessly as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends.

<p align="center">

https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c

</p>

**More advanced features will coming soon, so stay tuned!**

## 🚀 Quick Start

Follow these steps to get started with KTransformers:

### 📥 Installation

Install KTransformers using the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

We support:
*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

## 📃 Brief Injection Tutorial

KTransformers offers a template-based injection framework for easy integration of optimized modules.

<br/>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments with limited resources, with special attention to heterogeneous computing, such as GPU/CPU offloading of quantized models. For example, we support the efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Use YAML-based injection templates and call `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

`optimize_and_load_gguf` replaces model sub-modules based on rules defined in your YAML file.  The original `generate` interface is available. We also provide `prefill_and_generate` for further optimization.

### How to Custom your model

Find a detailed tutorial of injection and multi-GPU using DeepSeek-V2 as an example [here](doc/en/injection_tutorial.md).

Example YAML template for replacing all Linear modules with Marlin:

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

The `match` part specifies which module to replace, and `replace` specifies the module to inject, along with initialization keywords.

Example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 are in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. These power the `local_chat.py` demo.

For design principles and implementation details, see the [design document](doc/en/deepseek-v2-injection.md).

## 📝 Acknowledgment and Contributors

KTransformers builds upon the Transformers framework.  We appreciate the support of GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. We are planning to contribute back to the community by upstreaming our modifications.

KTransformers is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome new contributors!

## 💬 Discussion

Have questions?  Open an issue or join our WeChat group (QR code below).

<h2 id="FAQ">🙋 FAQ</h2>

Find answers to common questions in the [FAQ](doc/en/FAQ.md).