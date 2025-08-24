<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <br>
  <h1>KTransformers: Accelerate LLM Inference with Flexible Optimization</h1>
  <p>
     Supercharge your Hugging Face Transformers experience with KTransformers, a Python framework offering cutting-edge kernel optimizations for faster and more efficient LLM inference!
     <br>
     <a href="https://github.com/kvcache-ai/ktransformers"> ➡️  View the Original Repo</a>
  </p>
  <p align="center">
  <a href="#key-features">🌟 Key Features</a> | <a href="#showcases">🚀 Show Cases</a> | <a href="#quick-start">🏁 Quick Start</a> | <a href="#tutorial">📚 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a> | <a href="#faq">🙋 FAQ</a>
  </p>
</div>

## 🎉 Introduction

KTransformers is a flexible, Python-centric framework designed to drastically improve the performance of Large Language Models (LLMs) on your hardware. This powerful toolkit enhances your Hugging Face Transformers experience by seamlessly integrating advanced kernel optimizations and sophisticated placement/parallelism strategies. Easily implement and inject optimized modules with a single line of code, unlocking a Transformers-compatible interface, RESTful APIs (OpenAI and Ollama compatible), and even a simplified ChatGPT-like web UI.

## ✨ Key Features

*   **Accelerated Inference:** Leverage state-of-the-art kernel optimizations for significant speedups.
*   **Flexible & Extensible:** Python-centric design allows for easy integration of custom optimizations.
*   **Transformers Compatibility:** Maintain a familiar interface while benefiting from performance enhancements.
*   **RESTful API Support:**  Integrate with existing tools using OpenAI and Ollama-compatible APIs.
*   **Simplified UI:** Built-in web UI for easy interaction and testing.
*   **Heterogeneous Computing:** Efficiently utilize GPU/CPU offloading for quantized models
*   **Support for Latest Architectures:** Compatible with Intel Arc GPU, AMD and other popular architectures.

## 🔥 Recent Updates

Stay up-to-date with the latest improvements and model support:

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2 ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) prefix cache reuse ([Tutorial](./doc/en/prefix_cache.md))
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md))
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md))
*   **Apr 2, 2025:** Support Multi-concurrency ([Tutorial](./doc/en/balance-serve.md))
*   **Mar 15, 2025:** Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md))
*   **Mar 5, 2025:** Support unsloth 1.58/2.51 bits weights and IQ1_S/FP8 hybrid weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025:** Support FP8 GPU kernel for DeepSeek-V3 and R1; Longer Context.
*   **Feb 15, 2025:** Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update docs and online books.
*   **Feb 10, 2025:** Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup.
*   **Aug 28, 2024:** Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024:** Update detailed injection and multi-GPU tutorial
*   **Aug 14, 2024:** Support llamfile as linear backend.
*   **Aug 12, 2024:** Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024:** Support windows native.

## 🌟 Show Cases

KTransformers empowers you to run cutting-edge LLMs locally with impressive performance. See what's possible:

<div>
<h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

  <picture>
    <img alt="Local VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=50%>
  </picture>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, achieving results that surpass GPT-4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Performance Boost:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Seamlessly integrate as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends using an OpenAI and Ollama compatible API.

    <p align="center">
        <img alt="VSCode Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=50%>
    </p>

<!--
### 1M Context Local Inference on a Desktop with Only 24GB VRAM

<p align="center">
  <picture>
    <img alt="1M Context Local Inference" src="https://github.com/user-attachments/assets/a865e5e4-bca3-401e-94b8-af3c080e6c12" width=50%>
  </picture>

*   **1M Context InternLM 2.5 7B**: Run at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM.  Achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.

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

*   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels.  Over 10 times faster than llama.cpp's full attention approach.

*   **Flexible Sparse Attention Framework**:  Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm.  Further information is available [here](./doc/en/long_context_introduction.md).
-->

**More advanced features are coming soon!**

## 🏁 Quick Start

Get up and running with KTransformers quickly!

Supported Vendors: Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, AMD

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

## 📚 Brief Injection Tutorial

KTransformers utilizes a user-friendly, template-based injection framework.

<p align="center">
  <picture>
    <img alt="Injection Structure" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers excels in local deployments with limited resources, with a focus on heterogeneous computing, such as GPU/CPU offloading of quantized models. For example, we support the efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU, respectively. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

Integrate provided kernels by creating a YAML-based injection template and adding the call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

In this example, the AutoModel is initialized on the meta device to conserve memory. `optimize_and_load_gguf` then iterates through sub-modules, matches rules in your YAML file, and replaces them with advanced modules as specified.

The original `generate` interface is preserved, and we also provide a compatible `prefill_and_generate` method for further optimizations like CUDAGraph to boost generation speed.

### Customizing Your Model

A detailed tutorial on injection and multi-GPU use, using DeepSeek-V2 as an example, is available [here](doc/en/injection_tutorial.md).

Here's a YAML template example for replacing all original Linear modules with Marlin:

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

Each rule uses `match` to specify which modules to replace and `replace` to define the injected module and initialization keywords.

Find example rule templates for DeepSeek-V2 and Qwen2-57B-A14 optimization in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For more on the design principles and implementation of the injection framework, see the [design document](doc/en/deepseek-v2-injection.md).

##  🤝 Acknowledgment and Contributors

KTransformers builds upon the flexible Transformers framework and benefits from advanced kernels such as GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. The project aims to contribute modifications back to the community.

KTransformers is maintained by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/).  New contributors are welcome!

## 💬 Discussion

Ask questions or discuss KTransformers!

*   Open an issue.
*   Join our WeChat group: [WeChat Group](WeChatGroup.png)

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).