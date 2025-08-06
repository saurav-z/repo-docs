<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

# KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations

**KTransformers provides a flexible framework for accelerating Large Language Model (LLM) inference, enabling faster and more efficient deployments.  [Explore the KTransformers Repository](https://github.com/kvcache-ai/ktransformers)**

**[🌟 Show Cases](<#show-cases>) | [🚀 Quick Start](<#quick-start>) | [📃 Tutorial](<#tutorial>) | [💬 Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](<#FAQ>)**

## Key Features

*   **Seamless Integration:** Compatible with the Hugging Face Transformers library for easy integration.
*   **Kernel Optimization:** Includes advanced kernel optimizations for significant speedups.
*   **Flexible Framework:** Python-centric design allows for easy extension and customization.
*   **OpenAI/Ollama API Compatibility:**  Supports RESTful APIs compliant with OpenAI and Ollama for wider usability.
*   **Resource-Efficient:** Optimized for local deployments with limited resources.
*   **Cutting-Edge Support:**  Includes support for the latest models and quantization techniques.
*   **Multi-GPU Support:**  Leverages multi-GPU setups for enhanced performance.

## Updates

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))

https://github.com/user-attachments/assets/fafe8aec-4e22-49a8-8553-59fb5c6b00a2

*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025:** Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).

https://github.com/user-attachments/assets/faa3bda2-928b-45a7-b44f-21e12ec84b8a

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

### Local LLM Performance That Rivals GPT-4/o1

<p align="center">
  <picture>
    <img alt="GPT-4/o1-level Local VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=100%>
  </picture>
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed:**
        *   KTransformers: 54.21 tokens/s (32 cores) → 74.362 tokens/s (dual-socket, 2×32 cores) → 255.26 tokens/s (optimized AMX-based MoE kernel, V0.3 only) → 286.55 tokens/s (selectively using 6 experts, V0.3 only)
        *   Achieves up to **27.79x speedup** compared to llama.cpp (10.31 tokens/s).
    *   **Decode Speed:**
        *   KTransformers: 8.73 tokens/s (32 cores) → 11.26 tokens/s (dual-socket, 2×32 cores) → 13.69 tokens/s (selectively using 6 experts, V0.3 only)
        *   Achieves up to **3.03x speedup** compared to llama.cpp (4.51 tokens/s).
    *   **Coming Soon:** AMX optimizations and selective expert activation will be open-sourced in V0.3. Currently available in preview binary distribution.

*   **Local 236B DeepSeek-Coder-V2:** Runs the Q4\_K\_M version with only 21GB VRAM and 136GB DRAM, scoring better than GPT4-0613 in BigCodeBench.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

    *   **Faster Speed:**  126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation using MoE offloading and advanced kernels.
    *   **VSCode Integration:**  Integrated as a backend for Tabby and other frontends via an OpenAI and Ollama compatible API.

<p align="center">
  <picture>
    <img alt="VSCode Copilot Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=100%>
  </picture>
</p>

**More advanced features will be released soon!**

## 🚀 Quick Start

### Installation

Follow the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

### Supported Vendors

*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

## 📃 Brief Injection Tutorial

KTransformers uses a user-friendly, template-based injection framework. It lets you easily replace original torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments and efficient GPU/CPU offloading of quantized models, such as supporting efficient  <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels.

### Example Usage

Use YAML-based injection templates and call `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

After injection, the `generate` interface is available, with `prefill_and_generate` providing more optimizations.

### Customizing Your Model

See the [injection tutorial](doc/en/injection_tutorial.md) for a detailed example using DeepSeek-V2.

Here's an example YAML template:

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

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. See the [design document](doc/en/deepseek-v2-injection.md) for more details.

## <h2 id="ack">Acknowledgment and Contributors</h2>

KTransformers is built upon the Transformers framework and leverages advancements from GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.

KTransformers is actively maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## <h2 id="ack">Discussion

For questions, open an issue or join our WeChat group: [WeChat Group](WeChatGroup.png)

## <h2 id="FAQ">🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).