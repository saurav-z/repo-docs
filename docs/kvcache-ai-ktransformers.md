<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Kernel Optimizations</h1>
  <p><em>Unlock blazing-fast inference speeds and run cutting-edge LLMs on your local hardware with KTransformers!</em></p>
  
  <p>
  <strong><a href="#features">✨ Key Features</a> | <a href="#showcases">🚀 Show Cases</a> | <a href="#quick-start">🛠️ Quick Start</a> | <a href="#tutorial">📖 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a>|<a href="#faq"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers"> 🔗  View on GitHub</a></strong>
  </p>
</div>

## Overview

KTransformers is a Python-centric framework designed to accelerate your experience with 🤗 [Transformers](https://github.com/huggingface/transformers) through advanced kernel optimizations and strategic placement/parallelism techniques. Built for flexibility and ease of use, KTransformers empowers you to experiment with cutting-edge LLM inference optimizations.

## <a name="features">✨ Key Features</a>

*   **Enhanced Performance:** Achieve significant speedups in LLM inference, making local deployments more practical.
*   **Flexible Framework:** Easily integrate and experiment with optimized modules using a simple, Python-centric approach.
*   **Transformers Compatibility:** Seamlessly works with the Hugging Face Transformers ecosystem.
*   **OpenAI & Ollama API Compliance:** Supports RESTful APIs for easy integration with existing applications.
*   **User-Friendly Interface:** Includes a simplified ChatGPT-like web UI for intuitive interaction.
*   **Extensible Design:** Designed for easy integration of new optimization techniques.
*   **Hardware Support**: Support vendors such as Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, AMD, and others.
*   **Cutting-Edge Optimizations**: Support for various kernels like AMX-Int8, AMX-BF16, FP8 GPU kernel, and more.

## 🔥 Recent Updates

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

## <a name="showcases">🚀 Show Cases</a>

### GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM
<p align="center">
https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run its Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3.  Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run its Q4_K_M version using only 21GB VRAM and 136GB DRAM, attainable on a local desktop machine, which scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Wrapped into an OpenAI and Ollama compatible API for seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and various other frontends.

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
**More advanced features will coming soon, so stay tuned!**

## <a name="quick-start">🛠️ Quick Start</a>

Get started with KTransformers in a few simple steps!

We currently support the following vendors:

- Metax
- Sanechips (ZhuFeng V1.0)
- Intel
- Ascend
- Kunpeng
- AMD

### 📥 Installation

Refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for detailed instructions on installing KTransformers.

## <a name="tutorial">📖 Brief Injection Tutorial</a>

KTransformers employs a user-friendly, template-based injection framework, simplifying the process of incorporating optimized modules. This allows you to easily replace original PyTorch modules with optimized variants.

</br>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments with limited resources. It emphasizes heterogeneous computing, such as GPU/CPU offloading of quantized models. For instance, we support efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Utilize the provided kernels by creating a YAML-based injection template and adding a call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

Here, `AutoModel` initializes on the meta device. Then, `optimize_and_load_gguf` replaces sub-modules with advanced modules based on the YAML rules.

After injection, the original `generate` interface remains, and we also offer a `prefill_and_generate` method, enabling further optimizations like CUDAGraph for improved generation speed.

### Customizing Your Model

A detailed tutorial on injection and multi-GPU usage with DeepSeek-V2 can be found [here](doc/en/injection_tutorial.md).

Here's a YAML template example to replace Linear modules with Marlin, an advanced 4-bit quantization kernel:

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

Each YAML rule has `match` and `replace` sections. `match` specifies modules to replace, while `replace` defines the injected module and initialization keywords.

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. These templates are used in the `local_chat.py` demo.

For design principles and the injection framework implementation, see the [design document](doc/en/deepseek-v2-injection.md).

## <a name="ack">🤝 Acknowledgment and Contributors</a>

KTransformers builds upon the foundation of the Transformers library and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute our modifications back to the community.

KTransformers is developed and maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome new contributors!

## <a name="discussion">💬 Discussion</a>

For any questions, please open an issue. You can also join our WeChat group. QR Code: [WeChat Group](WeChatGroup.png)

## <a name="faq">🙋 FAQ</a>

Find answers to common questions in the [FAQ](doc/en/FAQ.md).