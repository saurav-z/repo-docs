<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h3>Supercharge Your LLM Inference: KTransformers Delivers Cutting-Edge Optimizations</h3>
  <p><i>Accelerate your Hugging Face Transformers experience with KTransformers!</i></p>

  <strong><a href="#key-features">✨ Key Features</a> | <a href="#showcases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a>|<a href="#faq"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">🔗 View on GitHub</a></strong>
</div>

## 🚀 Key Features

*   **Advanced Kernel Optimizations:** Experience significant speedups with specialized kernels.
*   **Flexible Framework:** Designed for easy extensibility and experimentation with cutting-edge LLM inference techniques.
*   **Transformers Compatibility:** Seamlessly integrates with Hugging Face Transformers.
*   **OpenAI & Ollama API Compliance:** Compatible with OpenAI and Ollama RESTful APIs.
*   **Simplified Web UI:** Includes a user-friendly, ChatGPT-like web UI.
*   **Heterogeneous Computing Support:** Optimized for GPU/CPU offloading and quantization.
*   **Multi-Vendor Support**: Optimized for vendors including Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.

## 🎉 Introduction

KTransformers, pronounced "Quick Transformers," is a Python-centric framework designed to enhance your experience with 🤗 [Transformers](https://github.com/huggingface/transformers) by providing advanced kernel optimizations and efficient placement/parallelism strategies. Build your own optimized LLM application with minimal code changes.

## 🔥 Updates

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

KTransformers unlocks impressive performance on consumer hardware, enabling advanced LLM functionalities locally.

### 💻 Local LLMs Powered by KTransformers:

<div>
<h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, which outperforms GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading using kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Integrated with OpenAI and Ollama compatible API for use with [Tabby](https://github.com/TabbyML/tabby) and other frontends.

<p align="center">

https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c

</p>

<!-- <h3>1M Context Local Inference on a Desktop with Only 24GB VRAM</h3>
<p align="center">

https://github.com/user-attachments/assets/a865e5e4-bca3-401e-94b8-af3c080e6c12

* **1M Context InternLM 2.5 7B**: Operates at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM, which is feasible on a local desktop setup. It achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.

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

* **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.

* **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
 -->

**More advanced features are coming soon!**

## 🚀 Quick Start

Get up and running with KTransformers quickly:

### 📥 Installation

Refer to the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for detailed instructions.

## 📃 Brief Injection Tutorial

KTransformers simplifies LLM optimization through its template-based injection framework.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers excels at local deployments by leveraging heterogeneous computing. It efficiently utilizes  [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU, respectively. More details [here](doc/en/operators/llamafile.md).

### Example Usage

Use YAML templates to inject kernels. The `optimize_and_load_gguf` call adds kernel optimization.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

Initialize the AutoModel with `meta` to avoid memory use, then `optimize_and_load_gguf` injects advanced modules using YAML rules.

The original `generate` interface is maintained, along with the optimized `prefill_and_generate` method.

### Custom Model Tutorial

See [here](doc/en/injection_tutorial.md) for a detailed tutorial on DeepSeek-V2.

Here is an example YAML template:

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

The YAML rules use a `match` part, and a `replace` part.

Find example rule templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For details about the design, see the [design document](doc/en/deepseek-v2-injection.md).

## <h2 id="ack">Acknowledgment and Contributors</h2>

KTransformers is built upon the Hugging Face Transformers library, and benefits from contributions from the GGUF/GGML, Llamafile, Marlin, sglang and flashinfer projects.

KTransformers is maintained by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome community contributions.

## <h2 id="ack">Discussion</h2>

Open an issue or join our [WeChat Group](WeChatGroup.png) for further discussion.

## <h2 id="FAQ">🙋 FAQ</h2>

Check the [FAQ](doc/en/FAQ.md) for answers to common questions.