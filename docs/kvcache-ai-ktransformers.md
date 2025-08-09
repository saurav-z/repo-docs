<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference</h1>
  <p><em>Unlock cutting-edge optimizations for Hugging Face Transformers, empowering faster and more efficient LLM inference with KTransformers.</em></p>

  <p>
    <a href="#features">🌟 Features</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#showcases">✨ Showcases</a> | <a href="#updates">🔥 Updates</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> | <a href="#faq">🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">🔗 Original Repo</a>
  </p>
</div>

## Overview

KTransformers, a flexible and extensible Python framework, revolutionizes your Hugging Face Transformers experience by integrating advanced kernel optimizations and strategic placement/parallelism techniques.  This open-source project allows users to easily experiment with and deploy cutting-edge LLM inference optimizations.  From enhanced speed to reduced VRAM usage, KTransformers unlocks the full potential of your local LLM deployments.

## <a name="features"></a>🌟 Key Features

*   **Simplified Optimization:** Inject optimized modules with a single line of code, seamlessly integrating with the Transformers API.
*   **OpenAI & Ollama Compatibility:** Supports both OpenAI and Ollama-compliant RESTful APIs, including a streamlined ChatGPT-like web UI.
*   **Heterogeneous Computing:** Leverages GPU/CPU offloading and quantization for resource-constrained environments.
*   **Extensible Architecture:**  Designed for experimentation and the easy integration of new optimization techniques.
*   **Performance Boost:** Achieves significant speedups and reduced VRAM usage, enabling powerful local LLM inference.

## <a name="updates"></a>🔥 Recent Updates

Stay up-to-date with the latest advancements in KTransformers:

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   ... (See the full changelog in the original README)

## <a name="showcases"></a>✨ Showcases

KTransformers delivers impressive performance improvements:

*   **Local GPT-4/o1-level VSCode Copilot:** Runs smoothly on a desktop with only 24GB VRAM.

    <p>
      <picture>
        <img alt="VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=70%>
      </picture>
    </p>

*   **Local 671B DeepSeek-Coder-V3/R1:** Achieves up to **27.79x speedup** in prefill and **3.03x speedup** in decode compared to llama.cpp, running the Q4_K_M version on only 14GB VRAM with 382GB DRAM. ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md))

    *   **Prefill Speed (tokens/s):**  Up to 286.55 (AMX-based MoE kernel with selective expert activation)
    *   **Decode Speed (tokens/s):** Up to 13.69 (selectively using 6 experts, V0.3 only)
*   **Local 236B DeepSeek-Coder-V2:** Runs the Q4_K_M version on a local desktop (21GB VRAM, 136GB DRAM), outperforming GPT-4 in BigCodeBench.

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

## <a name="quick-start"></a>🚀 Quick Start

### 📥 Installation

For detailed installation instructions, consult the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

### Supported Vendors

KTransformers supports a variety of hardware vendors:

*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

## Brief Injection Tutorial

KTransformers uses a template-based injection framework, enabling researchers to easily replace original torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Injection Structure" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

### Example Usage

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

### YAML Template Example

Example of a YAML template for replacing all original Linear modules with Marlin kernels:

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

## <a name="ack"></a>🤝 Acknowledgements and Contributions

KTransformers builds upon the powerful foundation of the Hugging Face Transformers library, and benefits from advancements in GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  The project is actively maintained by contributors from the MADSys group at Tsinghua University and members from Approaching.AI.

## Discussion

For any questions, please open an issue or join our WeChat group using the QR code: [WeChat Group](WeChatGroup.png)

## <a name="FAQ"></a>🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).