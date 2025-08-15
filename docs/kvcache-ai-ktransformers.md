<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <br>
  <h1>KTransformers: Supercharge Your LLM Inference with Advanced Optimizations</h1>
  <p><b>KTransformers</b> accelerates and optimizes your Hugging Face Transformers experience for faster and more efficient LLM inference. Explore the KTransformers repository on <a href="https://github.com/kvcache-ai/ktransformers">GitHub</a>.</p>

  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a> |
    <a href="#FAQ">🙋 FAQ</a>
  </p>
</div>

## Key Features

*   🚀 **Blazing-Fast Inference:** Leverage advanced kernel optimizations for significant speedups, achieving up to 27.79x faster prefill and 3.03x faster decode compared to llama.cpp.
*   🛠️ **Flexible and Extensible:**  A Python-centric framework designed for easy integration of new optimizations.
*   🧩 **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   🧠 **Optimized for Resource-Constrained Environments:**  Run large language models locally on desktop machines with limited VRAM and DRAM.
*   🌐 **OpenAI and Ollama API Compatibility:** Enables easy integration with existing LLM frontends and applications.
*   🔄 **Heterogeneous Computing Support:**  Optimized for GPU/CPU offloading and quantization to maximize hardware utilization.
*   🔥 **Cutting-Edge Kernel Integration:**  Supports advanced kernels like llamafile and Marlin for superior performance.

## 🚀 What's New

*   **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).

<!-- ... (previous updates) ... -->

## 🌟 Show Cases

*   **Local 671B DeepSeek-Coder-V3/R1:** Run its Q4_K_M version using only 14GB VRAM and 382GB DRAM.
    *   Achieves up to **27.79x speedup** in prefill and **3.03x speedup** in decode compared to llama.cpp.
    *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Runs its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM and scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).
*   **Integrated with VSCode:**  Wrapped into an OpenAI and Ollama compatible API for seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and various other frontends.

<!--  (Images and visual aids from the original README, optimized for SEO) -->

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

<!--
<p align="center">
  <picture>
    <img alt="Single Needle Retrieval 128K" src="./doc/assets/needle_128K.png" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <img alt="Single Needle Retrieval 1000K" src="./doc/assets/needle_1M.png" width=100%>
  </picture>
</p>
-->

## 🚀 Quick Start

Follow the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started with KTransformers.

## 📃 Brief Injection Tutorial

KTransformers uses a flexible, template-based injection framework to make it easy to replace original PyTorch modules with optimized ones.

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

### How to Custom Your Model

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

*   The `match` section specifies which modules to replace, and the `replace` section defines the optimized module, along with initialization parameters.
*   Find example templates in the `ktransformers/optimize/optimize_rules` directory.
*   Refer to the [design document](doc/en/deepseek-v2-injection.md) for more details on the injection framework.

## Acknowledgment and Contributors

KTransformers builds upon the foundations laid by the Transformers library and benefits from advanced kernels and contributions from the open-source community.

KTransformers is actively maintained and developed by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## Discussion

For questions and discussions, please open an issue on GitHub. You can also join our WeChat group (QR Code: [WeChatGroup.png]) for further discussion.

## 🙋 FAQ

Refer to the [FAQ](doc/en/FAQ.md) for answers to frequently asked questions.