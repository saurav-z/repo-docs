<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference</h1>
  <p><b>Experience cutting-edge LLM inference optimizations with KTransformers, a flexible framework for faster, more efficient AI.</b></p>
  <p>
    <a href="#features">🚀 Key Features</a> |
    <a href="#showcases">🌟 Show Cases</a> |
    <a href="#quickstart">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">📚 View on GitHub</a>
  </p>
</div>

## <a id="features"></a>🚀 Key Features

KTransformers empowers you to optimize your Hugging Face Transformers experience with advanced kernel optimizations, placement strategies, and parallelism.

*   **Flexible Framework:** Built with Python and designed for extensibility, allowing easy integration of optimized modules.
*   **Transformers Compatibility:** Provides a seamless interface compatible with existing Transformers workflows.
*   **OpenAI & Ollama API Compliance:** Supports RESTful APIs, making it easy to integrate with existing tools and applications.
*   **Simplified Web UI:** Includes a user-friendly web UI for a simplified ChatGPT-like experience.
*   **GPU/CPU Offloading:** Supports efficient GPU/CPU offloading of quantized models.
*   **Kernel Integration:** Supports optimized kernels like Llamafile and Marlin for CPU and GPU acceleration.
*   **Multi-Vendor Support:** Already supports vendors like Metax, Sanechips, Intel, Ascend, Kunpeng, and AMD, with more coming soon!

## 🔥 Updates
* **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
* **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
* **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
* **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
* **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))

* **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
* **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).

* **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
* **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
* **Feb 25, 2025**: Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
* **Feb 15, 2025**: Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
* **Feb 10, 2025**: Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
* **Aug 28, 2024**: Decrease DeepseekV2's required VRAM from 21G to 11G.
* **Aug 15, 2024**: Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
* **Aug 14, 2024**: Support llamfile as linear backend.
* **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
* **Aug 9, 2024**: Support windows native.

## <a id="showcases"></a>🌟 Show Cases

KTransformers enables high-performance LLM inference on resource-constrained hardware.

### GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

[Image of GPT-4/o1-level Local VSCode Copilot](https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285)

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   Prefill Speed (tokens/s):
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   Decode Speed (tokens/s):
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   Upcoming Open Source Release:
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Running its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, which scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

[Image of DeepSeek-Coder-V2 Score](https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693)

*   **Faster Speed:** Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
*   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via OpenAI and Ollama compatible API.

[Image of VSCode Integration](https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c)

<!--
### 1M Context Local Inference on a Desktop with Only 24GB VRAM

*   **1M Context InternLM 2.5 7B**: Operates at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM. It achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.

[Image of Single Needle Retrieval 128K](./doc/assets/needle_128K.png)

[Image of Single Needle Retrieval 1000K](./doc/assets/needle_1M.png)

*   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.

*   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
-->

**More advanced features are coming soon!**

## <a id="quickstart"></a>🚀 Quick Start

Get up and running with KTransformers in just a few steps!

### 📥 Installation

Follow the detailed [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

## <a id="tutorial"></a>📃 Brief Injection Tutorial

KTransformers features an easy-to-use, template-based injection framework for researchers to quickly experiment with optimizations.

[Image of Inject-Struction](https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e)

The injection framework allows you to replace original torch modules with optimized variants. This simplifies combining multiple optimizations.

KTransformers focuses on local deployments constrained by limited resources, and supports heterogeneous computing with GPU/CPU offloading and kernels such as <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a>. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Use a YAML-based injection template and add a call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

`optimize_and_load_gguf` iterates through all sub-modules of the model, matches rules specified in your YAML rule file, and replaces them with advanced modules. The original `generate` interface is available, but a compatible `prefill_and_generate` method is also provided to improve generation speed.

### How to customize your model

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

The `match` part specifies which module to replace, and the `replace` part specifies the injected module and initialization keywords.

Example rule templates can be found in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

Read the [design document](doc/en/deepseek-v2-injection.md) for design principles.

## <a id="ack"></a>Acknowledgment and Contributors

KTransformers builds on the flexibility of Transformers and benefits from advanced kernels. We plan to contribute back to the community.

KTransformers is developed by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  We welcome new contributors!

## <a id="discussion"></a>Discussion

Have questions? Open an issue or join our WeChat group: [WeChat Group](WeChatGroup.png)

## <a id="faq"></a>🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).