<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

# KTransformers: Supercharge Your LLM Inference with Optimized Kernels and Strategies

**KTransformers is a flexible framework that dramatically accelerates LLM inference, providing a powerful toolkit for researchers and developers.**  

[See the original repository](https://github.com/kvcache-ai/ktransformers)

**Key Features:**

*   🚀 **Enhanced Performance:** Achieve significant speedups for LLM inference through advanced kernel optimizations.
*   ⚙️ **Flexible Framework:** Easily integrate optimized modules with a single line of code, offering a Transformers-compatible interface.
*   🌐 **RESTful API & Web UI:** Includes RESTful APIs (OpenAI/Ollama compliant) and a simplified ChatGPT-like web UI for easy integration.
*   🛠️ **Extensible Design:**  Designed with extensibility at its core, enabling rapid experimentation with novel LLM inference strategies.
*   💪 **Support for Cutting-Edge Models:** Compatible with models like DeepSeek-Coder-V3/R1, LLaMA 4, Mixtral, and more.
*   🧠 **Optimized for Resource-Constrained Environments:**  Focuses on enabling high-performance LLM inference on local machines with limited resources.
*   💻 **Heterogeneous Computing Support:** Leverages heterogeneous computing, including GPU/CPU offloading for quantized models.

## What's New

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

## 🌟 Showcases

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Runs Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Performance:**
        *   Prefill Speed: Up to **27.79x speedup** compared to llama.cpp.
        *   Decode Speed: Up to **3.03x speedup** compared to llama.cpp.
    *   **Coming Soon:** AMX optimizations and selective expert activation will be open-sourced in V0.3.

*   **Local 236B DeepSeek-Coder-V2:** Runs its Q4_K_M version using only 21GB VRAM and 136GB DRAM, surpassing GPT-4-0613 in BigCodeBench.

    *   **Key Benefits:**
        *   **Faster Speed:** Prefill at 126 tokens/s (2K prompt), generate at 13.6 tokens/s.
        *   **VSCode Integration:** Compatible with Tabby and other frontends via OpenAI/Ollama API.

## 🚀 Quick Start

### 📥 Installation

For installation instructions, see the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Injection Tutorial

KTransformers uses a user-friendly, template-based injection framework to replace original Torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

*   Designed for local deployments, especially those with limited resources.
*   Supports heterogeneous computing, like GPU/CPU offloading with kernels such as <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a>.

**Example Usage:**

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

**Customizing Your Model:**

Use YAML templates to specify module replacements.

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

Find example rule templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.  See the [design document](doc/en/deepseek-v2-injection.md) for details.

## 🙏 Acknowledgment and Contributors

KTransformers is built upon the foundation of Transformers and benefits from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. The project is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and <a href="http://approaching.ai/">Approaching.AI</a>.  Contributions are welcome!

## 💬 Discussion

Have questions?  Open an issue or join our WeChat group (QR code in original README).

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).