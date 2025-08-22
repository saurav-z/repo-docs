<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h1>KTransformers: Supercharge Your LLM Inference with Advanced Kernel Optimizations</h1>
  <p><em>Unlock cutting-edge performance for your Hugging Face Transformers models with KTransformers, a flexible framework designed for blazing-fast inference on limited hardware.</em></p>

  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">🔗 View on GitHub</a>
  </p>
</div>

## Key Features

*   🚀 **Blazing-Fast Inference:** Experience significant speedups for LLMs through kernel optimizations and placement/parallelism strategies.
*   🛠️ **Flexible & Extensible:** Easily integrate optimized modules with a single line of code, maintaining compatibility with Hugging Face Transformers.
*   🧩 **Transformers-Compatible Interface:** Leverage existing Transformers workflows with minimal modifications.
*   🌐 **RESTful APIs:** Supports OpenAI and Ollama-compliant APIs for seamless integration.
*   💻 **Simplified UI:** Includes a ChatGPT-like web UI for easy experimentation.
*   💡 **Heterogeneous Computing:** Optimized for GPU/CPU offloading of quantized models.
*   ⚙️ **Model Agnostic:** Supports a wide range of models, including DeepSeek, Llama, and more.

## 🔥 What's New

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
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

<!-- * **Aug 28, 2024**: Support 1M context under the InternLM2.5-7B-Chat-1M model, utilizing 24GB of VRAM and 150GB of DRAM. The detailed tutorial is [here](./doc/en/long_context_tutorial.md). -->

## 🌟 Show Cases

### GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

[Demo Video](https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285)

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).

*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, achieving performance surpassing GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
    </p>

    *   **Faster Speed:** Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Wrapped into an OpenAI and Ollama compatible API for seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and various other frontends.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" />
    </p>

<!--
### 1M Context Local Inference on a Desktop with Only 24GB VRAM

<p align="center">
  <img src="https://github.com/user-attachments/assets/a865e5e4-bca3-401e-94b8-af3c080e6c12" />
</p>

*   **1M Context InternLM 2.5 7B**: Operates at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM, which is feasible on a local desktop setup. It achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.

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

*   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.

*   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
 -->

**More advanced features are coming soon!**

## 🚀 Quick Start

Follow these steps to get started with KTransformers:

### 📥 Installation

Refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for detailed instructions.

## 📃 Brief Injection Tutorial

KTransformers employs a user-friendly, template-based injection framework, empowering researchers to effortlessly substitute original torch modules with optimized variants. This simplifies the process of combining multiple optimizations, enabling the exploration of their synergistic effects.

<p align="center">
  <img alt="Injection Process" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</p>

KTransformers focuses on local deployments constrained by limited resources, focusing on heterogeneous computing opportunities like GPU/CPU offloading of quantized models. It supports efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Integrate provided kernels by creating a YAML-based injection template and adding the call to `optimize_and_load_gguf` before using the Transformers model:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

This example initializes AutoModel on the meta device to avoid memory usage. `optimize_and_load_gguf` then iterates through sub-modules, matches rules specified in your YAML rule file, and replaces them with advanced modules.

The original `generate` interface is available after injection, but a compatible `prefill_and_generate` method is also provided for further optimizations like CUDAGraph.

### How to Customize Your Model

A detailed tutorial for injection and multi-GPU usage, using DeepSeek-V2 as an example, is available [here](doc/en/injection_tutorial.md).

Example YAML template for replacing all original Linear modules with Marlin (4-bit quantization):

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

Each rule in the YAML file includes `match` and `replace` parts. `match` specifies which module to replace and `replace` specifies the module to be injected.

Example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 are available in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory, powering the `local_chat.py` demo.

For detailed design principles and the injection framework implementation, refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🙏 Acknowledgment and Contributors

KTransformers builds upon the flexibility of Hugging Face Transformers, benefiting from kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute modifications back to the community.

KTransformers is actively maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. New contributors are welcome to join us in enhancing KTransformers.

## 🙋 Discussion

For questions, please open an issue or join our WeChat group (QR Code below).

<img src="WeChatGroup.png" alt="WeChat Group QR Code" width="200"/>

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).