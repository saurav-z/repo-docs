<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h3>Supercharge Your LLM Experience with KTransformers: Optimized Inference, Simplified.</h3>
  <strong><a href="#key-features">🚀 Key Features</a> | <a href="#showcases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a>|<a href="#faq"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">🔗 Original Repo</a> </strong>
</div>

## Key Features

*   **Enhanced Performance:** Experience significant speedups in Large Language Model (LLM) inference using advanced kernel optimizations.
*   **Flexible Framework:** Easily integrate and experiment with cutting-edge inference techniques through a Python-centric design.
*   **Transformers Compatibility:** Seamlessly works with Hugging Face Transformers, maintaining a familiar interface.
*   **OpenAI/Ollama API Support:** Supports RESTful APIs compatible with OpenAI and Ollama for easy integration with existing tools.
*   **Simplified Web UI:** Includes a basic ChatGPT-like web UI for quick experimentation.
*   **Heterogeneous Computing:** Leverage GPU/CPU offloading for optimized performance on limited resources, with support for Llamafile and Marlin kernels.
*   **Model Support:** Supports a wide range of models, including DeepSeek-Coder, Qwen, and Llama models.
*   **Active Development:** Benefit from ongoing updates and new features to stay ahead of the curve.

## What is KTransformers?

KTransformers, or "Quick Transformers," is a Python-based framework designed to accelerate your LLM inference using advanced kernel optimizations, placement, and parallelism strategies. It offers a flexible and extensible platform for researchers and developers to experiment with innovative inference techniques.  [Learn more on GitHub](https://github.com/kvcache-ai/ktransformers).

## 🔥 Updates

*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8, AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
*   **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
*   **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and [IQ1\_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
*   **Feb 25, 2025**: Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
*   **Feb 15, 2025**: Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
*   **Feb 10, 2025**: Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Aug 28, 2024**: Decrease DeepseekV2's required VRAM from 21G to 11G.
*   **Aug 15, 2024**: Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
*   **Aug 14, 2024**: Support llamfile as linear backend.
*   **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
*   **Aug 9, 2024**: Support windows native.

## <a id="showcases"></a> 🌟 Show Cases

### Achieve GPT-4/o1-Level Performance on Your Desktop (24GB VRAM)

  - **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    -   **Prefill Speed (tokens/s):**
        -   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        -   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    -   **Decode Speed (tokens/s):**
        -   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        -   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    -   **Upcoming Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3. Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).

  -   **Local 236B DeepSeek-Coder-V2:** Run its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, achieving performance that rivals GPT-4-0613, as shown on the [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

  -   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
  -   **VSCode Integration:** Seamlessly integrate with [Tabby](https://github.com/TabbyML/tabby) and other frontends by wrapping KTransformers in an OpenAI and Ollama-compatible API.

<p align="center">
https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c
</p>

### 1M Context Local Inference on a Desktop with Only 24GB VRAM

<!--
<p align="center">
https://github.com/user-attachments/assets/a865e5e4-bca3-401e-94b8-af3c080e6c12
</p>
-->
<!--
*   **1M Context InternLM 2.5 7B:** Operates at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM, which is feasible on a local desktop setup. It achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.

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
<!--
*   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.

*   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
-->

**More advanced features are coming soon, so stay tuned!**

## <a id="quick-start"></a> 🚀 Quick Start

### 📥 Installation

To get started with KTransformers, follow these steps:

1.  **Install KTransformers:** Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).
2.  **Install Dependencies:** Make sure you have all required dependencies installed based on your chosen features.

## <a id="tutorial"></a> 📃 Brief Injection Tutorial

KTransformers uses a user-friendly, template-based injection framework to enhance your 🤗 Transformers experience.  This approach simplifies the replacement of standard torch modules with optimized variants. It also streamlines the process of combining optimizations, allowing for the exploration of their synergistic effects.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

### Example Usage

Here's how to utilize the provided kernels:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

In this example, the AutoModel is first initialized on the meta device to avoid memory usage. Then, `optimize_and_load_gguf` iterates through sub-modules, matches rules in your YAML file, and replaces them with optimized modules.  After injection, the original `generate` interface is available, as well as the `prefill_and_generate` method, which enables optimizations like CUDAGraph.

### Customizing Your Model

A detailed tutorial for injection and multi-GPU use, using DeepSeek-V2 as an example, is available [here](doc/en/injection_tutorial.md).

Here is a YAML template for replacing all original Linear modules with Marlin, an advanced 4-bit quantization kernel:

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

Each rule in the YAML file has two parts: `match` and `replace`. The `match` part specifies which module should be replaced, and the `replace` part specifies the module to be injected along with initialization keywords.

Find example rule templates for DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory, used in the `local_chat.py` demo. For design principles and the implementation of the injection framework, refer to the [design document](doc/en/deepseek-v2-injection.md).

## <a id="ack"></a> Acknowledgment and Contributors

KTransformers is built upon the foundations of Transformers, leveraging advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  We plan to contribute modifications back to the community.

KTransformers is maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and <a href="http://approaching.ai/">Approaching.AI</a>.  New contributors are welcome to join the project!

## <a id="discussion"></a> Discussion

For questions, open an issue.  You can also join our WeChat group (QR code below) for discussions.

<p align="center">
  <img src="WeChatGroup.png" alt="WeChat Group QR Code" width="150"/>
</p>

## <a id="FAQ"></a> 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).