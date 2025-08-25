<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

## KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations

**KTransformers, a flexible Python framework, accelerates your Hugging Face Transformers experience with advanced kernel optimizations and efficient placement/parallelism strategies, enabling faster and more efficient LLM inference. [Explore the KTransformers Repository](https://github.com/kvcache-ai/ktransformers)**

**[🌟 Show Cases](#show-cases) | [🚀 Quick Start](#quick-start) | [📃 Tutorial](#tutorial) | [💬  Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](#FAQ)**

---

## Key Features

*   **Optimized Inference:** Leverages advanced kernel optimizations for significant speedups in LLM inference.
*   **Flexible Framework:** Designed for extensibility, allowing easy integration of custom optimizations.
*   **Transformers Compatibility:** Provides a seamless Transformers-compatible interface.
*   **RESTful APIs:** Includes RESTful APIs compatible with OpenAI and Ollama for easy integration.
*   **Simplified Web UI:** Offers a streamlined, ChatGPT-like web UI.
*   **Heterogeneous Computing Support:** Explores GPU/CPU offloading of quantized models.
*   **Active Development:** Constantly updated with new features and optimizations.

## 🔥 Recent Updates

Stay up-to-date with the latest advancements:

*   **July 26, 2025:** Added support for SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Added support for Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025:** Added support for Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Added support for AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **(Older updates are in the original README.)**

## 🌟 Show Cases

KTransformers delivers impressive performance improvements, demonstrated through compelling use cases:

### Local LLM Inference

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Runs the Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, achieving performance that surpasses GPT4-0613 in the [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Faster Speed:** Reaching 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via OpenAI and Ollama compatible API.

    <p align="center">
        <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" alt="VSCode Integration" width=75%>
    </p>

## 🚀 Quick Start

Get up and running with KTransformers quickly!

### 📥 Installation

Refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for detailed instructions.

## 📃 Brief Injection Tutorial

KTransformers uses a template-based injection framework, making it easy to inject optimized modules into your Transformer models.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments constrained by resources, paying attention to heterogeneous computing opportunities, such as GPU/CPU offloading of quantized models. For example, we support the efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

To use the provided kernels, create a YAML-based injection template and add the call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

This example initializes the AutoModel on the meta device and then uses `optimize_and_load_gguf` to replace modules with optimized variants.  The original `generate` interface is available, and a `prefill_and_generate` method is provided for further optimization (like CUDAGraph) to improve generation speed.

### Customizing Your Model

A detailed tutorial for injection and multi-GPU usage, using DeepSeek-V2 as an example, is available [here](doc/en/injection_tutorial.md).

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

Each rule in the YAML file has a `match` and a `replace` part. The `match` specifies the modules to be replaced, and the `replace` specifies the replacement module with initialization keywords.

Example rule templates can be found in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

Refer to the [design document](doc/en/deepseek-v2-injection.md) for more details on the injection framework.

## 🤝 Acknowledgments and Contributors

KTransformers is built upon the Transformers framework and benefits from the work of GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. Development is led by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  Contributions are welcome!

## 🙋 FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).