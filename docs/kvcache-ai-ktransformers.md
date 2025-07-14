<div align="center">
    <picture>
        <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
    </picture>
</div>

# KTransformers: Supercharge Your LLM Inference with Optimized Kernels

**KTransformers empowers you to run large language models (LLMs) faster and more efficiently, optimized for local deployments.** [Explore the KTransformers Repository](https://github.com/kvcache-ai/ktransformers)

**Key Features:**

*   🚀 **Blazing Fast Inference:** Experience significant speedups for LLMs using advanced kernel optimizations and placement/parallelism strategies.
*   🧩 **Flexible Framework:** Designed with extensibility, allowing for easy integration of optimized modules with a single line of code.
*   💻 **Hugging Face Transformers Compatible:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   ⚙️ **OpenAI & Ollama API Compatibility:** Provides RESTful APIs that are compliant with OpenAI and Ollama for easy integration with existing tools.
*   💡 **Simplified Web UI:** Includes a simplified ChatGPT-like web UI for easy interaction.
*   🔄 **Multi-Platform Support:**  Includes support for a variety of hardware vendors, including Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.
*   🤝 **Community Driven:** Actively developed by the MADSys group at Tsinghua University and members from Approaching.AI, with open-source contributions planned.

## Key Updates & Recent Developments

*   **July 11, 2025:** Support Kimi-K2.
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

### Local LLM Performance on Desktop with Limited Resources

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Speedup Highlights (vs. llama.cpp with 2×32 cores):**
        *   Prefill: Up to **27.79x speedup** (AMX optimized MoE kernel).
        *   Decode: Up to **3.03x speedup**.
    *   **Upcoming Open Source:** AMX optimizations and selective expert activation will be open-sourced in V0.3.  (Preview binary available [here](./doc/en/DeepseekR1_V3_tutorial.md))

*   **Local 236B DeepSeek-Coder-V2:** Achieves performance exceeding GPT-4-0613 in BigCodeBench while running the Q4\_K\_M version with only 21GB VRAM and 136GB DRAM on a local desktop.

    <p align="center">
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
    </p>

    *   **Key Features:**
        *   **Faster Speed:** Up to 126 tokens/s for 2K prompt prefill and 13.6 tokens/s generation via MoE offloading using advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
        *   **VSCode Integration:** Supports OpenAI and Ollama compatible API integration for Tabby and other frontends.

    <p align="center">
        <img alt="VSCode Copilot Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c">
    </p>

More advanced features are coming soon.

## 🚀 Quick Start

### 📥 Installation

Refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for detailed installation instructions.

## 📃 Injection Tutorial

KTransformers offers a user-friendly, template-based injection framework, enabling easy module replacement with optimized variants.

<p align="center">
  <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</p>

KTransformers focuses on local deployments, especially those with limited resources, including heterogeneous computing opportunities like GPU/CPU offloading of quantized models using [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels.

### Example Usage

Use YAML templates to replace modules and call `optimize_and_load_gguf`:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

### Customizing Your Model

A detailed injection and multi-GPU tutorial using DeepSeek-V2 as an example is provided [here](doc/en/injection_tutorial.md).

YAML template example:

```yaml
- match:
    name: "^model\\.layers\\..*$"
    class: torch.nn.Linear
  replace:
    class: ktransformers.operators.linear.KTransformerLinear
    device: "cpu"
    kwargs:
      generate_device: "cuda"
      generate_linear_type: "QuantizedLinearMarlin"
```

Find example rule templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

## 🤝 Acknowledgments and Contributors

KTransformers builds upon the Transformers framework and benefits from advanced kernels such as GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  It is actively maintained by the MADSys group at Tsinghua University and members from Approaching.AI.

## 💬 Discussion

For any questions, open an issue or join our WeChat group (QR code below):

<img src="WeChatGroup.png" alt="WeChat Group QR Code" width="100">

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).