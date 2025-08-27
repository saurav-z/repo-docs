<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Optimized Kernels</h1>
  <p><em>Unlock lightning-fast inference speeds and run cutting-edge LLMs efficiently on your hardware.</em></p>
  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">🔗 Original Repo</a>
  </p>
</div>

## Key Features

*   🚀 **Accelerated Inference:** Experience significant speedups in LLM inference using optimized kernels.
*   🛠️ **Flexible Framework:** Easily integrate and experiment with advanced kernel optimizations for Transformers models.
*   🧩 **Hugging Face Transformers Compatibility:** Seamlessly works with the Hugging Face Transformers ecosystem.
*   🌐 **OpenAI & Ollama API Compatibility:** Supports RESTful APIs compatible with OpenAI and Ollama.
*   💻 **Local Deployment Focus:** Optimized for local deployments, maximizing performance on resource-constrained hardware.
*   🔄 **Multi-GPU Support:** Leverage multiple GPUs for increased performance.
*   🔬 **Extensible Architecture:** Designed for easy extension and integration of new optimization techniques.

## What is KTransformers?

KTransformers is a Python-centric framework designed to revolutionize your Hugging Face Transformers experience. This powerful tool enhances LLM inference with advanced kernel optimizations and flexible placement/parallelism strategies. With KTransformers, you can easily inject optimized modules into your models, access a Transformers-compatible interface, and enjoy features like RESTful APIs compliant with OpenAI and Ollama, and even a simplified ChatGPT-like web UI.

## 🔥 Recent Updates

*   **July 26, 2025:** Support for SmallThinker and GLM4-MoE ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md)).
*   **July 11, 2025:** Support for Kimi-K2 ([Tutorial](./doc/en/Kimi-K2.md)).
*   **June 30, 2025:** Support for 3-layer (GPU-CPU-Disk) prefix cache reuse.
*   **May 14, 2025:** Support for Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support for AMX-Int8, AMX-BF16, and Qwen3MoE ([Tutorial](./doc/en/AMX.md)).
*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025:** Support for Multi-concurrency ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025:** Support for ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025:** Support unsloth 1.58/2.51 bits weights and IQ1_S/FP8 hybrid weights; Support 139K Longer Context for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025:** Support FP8 GPU kernel for DeepSeek-V3 and R1; Longer Context.
*   **Feb 15, 2025:** Longer Context (from 4K to 8K for 24GB VRAM) & Faster Speed; update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025:** Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup.

## 🌟 Show Cases

### Local LLM Magic: Real-World Performance Boosts

KTransformers unlocks incredible LLM performance on your local hardware.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4_K_M version with just 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Impressive Speedups:**
        *   **Prefill:** Up to 27.79x faster than llama.cpp.
        *   **Decode:** Up to 3.03x faster than llama.cpp.
    *   AMX Optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Run its Q4_K_M version on a local desktop, exceeding GPT-4-0613 in BigCodeBench!

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Performance:**
        *   126 tokens/s for 2K prompt prefill.
        *   13.6 tokens/s for generation.
    *   **Features:**
        *   MoE offloading and advanced kernel injection from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
        *   OpenAI/Ollama compatible API for integration with Tabby and other frontends.

## 🚀 Quick Start

Get up and running with KTransformers quickly.

### 📥 Installation

Install KTransformers by following the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Brief Injection Tutorial

KTransformers employs a user-friendly, template-based injection framework, making it easy to replace original torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Injection Process" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

By leveraging the strengths of frameworks like vLLM, KTransformers focuses on optimizing for local deployments with limited resources, paying special attention to heterogeneous computing (GPU/CPU offloading, quantized models), e.g. efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

1.  **Initialize Model:**  First initialize the model on the meta device to avoid occupying memory.
2.  **Optimize and Load:**  Call `optimize_and_load_gguf`
3.  **Generate:**  Use the original `generate` interface or our optimized  `prefill_and_generate` method.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

### Customizing Your Model

Create YAML-based injection templates to customize your model optimizations.  A detailed tutorial is available [here](doc/en/injection_tutorial.md).

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

## Acknowledgment and Contributors

KTransformers builds upon the foundation of the Hugging Face Transformers library and benefits from contributions from GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. It is actively developed by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome new contributions!

## Discussion

Have questions?  Open an issue!  Join our WeChat group for more discussion.  QR Code: [WeChat Group](WeChatGroup.png)

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).