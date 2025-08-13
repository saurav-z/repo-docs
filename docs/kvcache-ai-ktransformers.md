<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <p><b>KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations</b></p>
  <p>Enhance your Hugging Face Transformers experience with advanced kernel optimizations and placement/parallelism strategies for faster and more efficient LLM inference.  <a href="https://github.com/kvcache-ai/ktransformers">Explore KTransformers on GitHub!</a></p>
  <br>
  <p>
    <a href="#showcases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a>
  </p>
</div>

## Key Features

*   **Optimized LLM Inference:**  Achieve significant speedups in LLM inference through kernel optimizations, model placement, and parallelism.
*   **Flexible Framework:**  Easily experiment with different optimization techniques using a Python-centric framework designed for extensibility.
*   **Transformers Compatibility:**  Seamlessly integrate with the Hugging Face Transformers ecosystem.
*   **RESTful APIs:** Compatible with OpenAI and Ollama APIs, for easy integration with various frontends.
*   **Simplified Web UI:** Quickly deploy a ChatGPT-like web UI for model interaction.
*   **Support for Diverse Hardware:**  Optimized for various hardware vendors, including Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.
*   **Advanced Kernel Injection:** Easily integrate optimized modules with a simple YAML-based configuration.
*   **Long Context Support:** Optimized support for extremely long contexts.

## 🔥 Updates

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

## <h2 id="showcases">🌟 Show Cases</h2>

KTransformers unlocks impressive performance improvements, enabling powerful LLM experiences even on resource-constrained hardware.

*   **Local 671B DeepSeek-Coder-V3/R1:**  Run the Q4_K_M version using only 14GB VRAM and 382GB DRAM. ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Significant Speedups:**
        *   **Prefill:** Up to 27.79x faster than llama.cpp (2×32 cores).
        *   **Decode:** Up to 3.03x faster than llama.cpp (2×32 cores).
    *   **Optimizations:** Includes AMX optimizations and selective expert activation (V0.3 preview binary available [here](./doc/en/DeepseekR1_V3_tutorial.md)).
*   **Local 236B DeepSeek-Coder-V2:**  Run the Q4_K_M version with only 21GB VRAM and 136GB DRAM on a local desktop, outperforming GPT4-0613 in BigCodeBench.
    *   **High Performance:**  Achieves 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    *   **VSCode Integration:** Compatible with Tabby and other frontends via OpenAI/Ollama API.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **GPT-4/o1-level Local VSCode Copilot:** Enables GPT-4/o1-level local VSCode Copilot on a desktop with only 24GB VRAM.

<p align="center">

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

</p>
**More advanced features will coming soon, so stay tuned!**

## <h2 id="quick-start">🚀 Quick Start</h2>

Get up and running with KTransformers quickly.

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started.

## <h2 id="tutorial">📃 Brief Injection Tutorial</h2>

KTransformers uses a template-based injection framework to make it simple for researchers to replace torch modules with optimized variants and combine optimizations.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments constrained by limited resources, especially on heterogeneous computing opportunities, like GPU/CPU offloading of quantized models.  Supports <a href="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a href="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a href="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Use YAML-based injection templates with `optimize_and_load_gguf`.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

`optimize_and_load_gguf` replaces sub-modules with advanced modules, and the original `generate` interface is available. A `prefill_and_generate` method provides further optimizations.

### How to Customize Your Model

A detailed tutorial for injection and multi-GPU using DeepSeek-V2 as an example is given [here](doc/en/injection_tutorial.md).

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

The `match` section specifies modules to replace, and the `replace` section defines the injected module and its initialization.

Find example rule templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.  See the [design document](doc/en/deepseek-v2-injection.md) for more details on the injection framework.

## <h2 id="ack">Acknowledgment and Contributors</h2>

KTransformers leverages Transformers and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer.  The project is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## <h2 id="ack">Discussion</h2>

Join the discussion!  Open an issue or join our WeChat group (QR Code: [WeChat Group](WeChatGroup.png)).

## <h2 id="FAQ">🙋 FAQ</h2>

Find answers to common questions in the [FAQ](doc/en/FAQ.md).