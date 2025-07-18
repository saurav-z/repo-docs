<div align="center">
    <picture>
        <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
    </picture>
</div>

## KTransformers: Supercharge Your LLM Inference with Advanced Optimizations

**KTransformers is a flexible framework designed to enhance your Hugging Face Transformers experience, providing cutting-edge kernel optimizations and placement/parallelism strategies for faster and more efficient LLM inference.**

[**View the original repo on GitHub**](https://github.com/kvcache-ai/ktransformers)

**[🌟 Show Cases](#show-cases) | [🚀 Quick Start](#quick-start) | [📃 Tutorial](#tutorial) | [💬  Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](#FAQ)**

---

## Key Features

*   **Flexible and Extensible:** Easily integrate optimized modules with a single line of code for a Transformers-compatible interface.
*   **OpenAI & Ollama Compatibility:** Supports RESTful APIs compliant with OpenAI and Ollama standards.
*   **Simplified Web UI:** Includes a user-friendly, ChatGPT-like web UI for easy interaction.
*   **Heterogeneous Computing Support:** Leverages GPU/CPU offloading for quantized models.
*   **Advanced Kernel Integration:** Integrates cutting-edge kernels like Llamafile and Marlin for improved performance.
*   **Multi-Platform Support:** Native Windows support and ROCm on AMD GPUs.
*   **Cutting-Edge Optimizations:** Includes support for AMX-Int8, AMX-BF16, Qwen3MoE, and experimental LLaMA 4 models.

---

## 🔥 Updates

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

---

## 🌟 Show Cases

KTransformers delivers significant performance improvements for LLMs.

*   **Local 671B DeepSeek-Coder-V3/R1:** Runs its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Prefill Speed:** Up to 286.55 tokens/s with optimized AMX-based MoE kernel.  Achieves up to **27.79× speedup** compared to llama.cpp.
    *   **Decode Speed:** Up to 13.69 tokens/s. Achieves up to **3.03× speedup** compared to llama.cpp.
    *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Runs its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, exceeding GPT4-0613 in BigCodeBench.
    *   **Faster Speed:** 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    *   **VSCode Integration:** Integrated as a backend for Tabby and other frontends via OpenAI and Ollama compatible API.

---

## 🚀 Quick Start

KTransformers is easy to set up.

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started.

---

## 📃 Brief Injection Tutorial

KTransformers uses a template-based injection framework for easy module optimization.

<div align="center">
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</div>

The framework focuses on local deployments with limited resources, with a focus on GPU/CPU offloading of quantized models.
For example, we support the efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU, respectively.

### Example Usage

Use YAML templates to replace original torch modules with optimized variants by calling  `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

### How to Customize Your Model

A detailed tutorial of the injection and multi-GPU using DeepSeek-V2 as an example is given [here](doc/en/injection_tutorial.md).

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

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

Refer to the [design document](doc/en/deepseek-v2-injection.md) for more details on the injection framework's design.

---

## Acknowledgment and Contributors

KTransformers is built upon the Transformers framework and benefits from the contributions of projects such as GGUF/GGML, Llamafile, Marlin, sglang and flashinfer.

KTransformers is actively maintained by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. Contributions are welcome!

---

## Discussion

For questions, open an issue. Join our WeChat group via the QR Code:
[WeChat Group](WeChatGroup.png)

---

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).