<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Optimized Kernels</h1>
  <p>KTransformers provides a flexible framework for accelerating Large Language Model (LLM) inference, unlocking cutting-edge performance on your hardware. <a href="https://github.com/kvcache-ai/ktransformers">Explore the KTransformers repository!</a></p>
  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a>
  </p>
</div>

## 🎉 Introduction

KTransformers, or "Quick Transformers," is a Python-centric framework designed to dramatically improve the performance of 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers). It offers a streamlined way to integrate advanced kernel optimizations, placement strategies, and parallelism techniques into your LLM workflows.  With KTransformers, you can easily leverage optimized modules, enjoy a Transformers-compatible interface, and even deploy RESTful APIs compatible with OpenAI and Ollama, along with a simplified ChatGPT-like web UI.

**Key Features:**

*   **Optimized Kernel Injection:** Easily integrate cutting-edge optimization kernels with a single line of code.
*   **Transformers Compatibility:**  Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   **OpenAI/Ollama API Support:**  Deploy models with compatible RESTful APIs.
*   **Flexible & Extensible:** Built with extensibility in mind, allowing for easy experimentation with new optimization strategies.
*   **Hardware Support:** Supports various vendors include Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.

## 🔥 Updates

*   **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))

    https://github.com/user-attachments/assets/fafe8aec-4e22-49a8-8553-59fb5c6b00a2

*   **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).

    https://github.com/user-attachments/assets/faa3bda2-928b-45a7-b44f-21e12ec84b8a

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

<!-- * **Aug 28, 2024**: Support 1M context under the InternLM2.5-7B-Chat-1M model, utilizing 24GB of VRAM and 150GB of DRAM. The detailed tutorial is [here](./doc/en/long_context_tutorial.md). -->

## 🌟 Show Cases

### 🚀 Local LLMs Powering Cutting-Edge Applications

KTransformers empowers you to run powerful LLMs locally, enabling applications like:

*   **GPT-4/o1-level Local VSCode Copilot:** Experience a responsive and powerful coding assistant on your desktop with limited VRAM (24GB).

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

*   **[NEW!!!] 671B DeepSeek-Coder-V3/R1 Locally:** Run its Q4_K_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).

*   **236B DeepSeek-Coder-V2 Locally:** Run the Q4_K_M version using only 21GB VRAM and 136GB DRAM, achieving performance exceeding GPT-4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

    *   **Faster Speed:**  Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Integrated with OpenAI and Ollama-compatible API for seamless use as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends.

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

**More advanced features are coming soon, stay tuned!**

## 🚀 Quick Start

Get up and running with KTransformers quickly!

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

## 📃 Brief Injection Tutorial

KTransformers simplifies the process of replacing torch modules with optimized variants via a user-friendly template-based injection framework. This allows you to combine optimizations and explore their synergistic effects.

</br>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments constrained by limited resources and addresses heterogeneous computing opportunities, especially GPU/CPU offloading of quantized models. This framework efficiently supports <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Utilize provided kernels by creating a YAML-based injection template and adding a call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

The `optimize_and_load_gguf` function iterates through all sub-modules of the model, matches rules in your YAML rule file, and replaces them with advanced modules. After injection, the original `generate` interface is available, along with `prefill_and_generate` for additional optimizations.

### How to Customize Your Model

See a detailed tutorial of the injection and multi-GPU using DeepSeek-V2 as an example [here](doc/en/injection_tutorial.md).

YAML template example for replacing all original Linear modules with Marlin:

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

Each rule in the YAML file contains `match` and `replace`. `match` specifies which module to replace, and `replace` specifies the module to inject, along with initialization keywords.

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

Refer to the [design document](doc/en/deepseek-v2-injection.md) for more information on the injection framework's design principles and implementation.

## <a id="ack"></a>Acknowledgment and Contributors

KTransformers leverages the flexibility of the Transformers framework, along with advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute back to the community.

KTransformers is actively maintained and developed by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## <a id="discussion"></a>Discussion

For questions, open an issue or join our WeChat group: [WeChat Group](WeChatGroup.png)

## <a id="FAQ"></a>🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).