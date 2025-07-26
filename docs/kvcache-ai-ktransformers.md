<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

## KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations

**KTransformers revolutionizes your experience with Hugging Face Transformers, providing a flexible framework for integrating advanced kernel optimizations.** Explore cutting-edge techniques and achieve significant performance gains with your favorite language models.  [Explore KTransformers on GitHub](https://github.com/kvcache-ai/ktransformers)

**Key Features:**

*   🚀 **Flexible Framework:** Designed for extensibility, making it easy to experiment with and integrate new optimizations.
*   ⚡ **Kernel Optimization:** Leverage advanced kernels for faster inference speeds.
*   🧠 **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   🌐 **API & UI Support:** Provides Transformers-compatible interface, RESTful APIs (OpenAI & Ollama compliant), and a simplified web UI for easy access.
*   💻 **Heterogeneous Computing:** Optimizes for GPU/CPU offloading, leveraging the power of multiple devices.
*   🧪 **Experimentation:** Ideal for researchers and developers to explore innovative LLM inference optimizations.
*   🛠️ **Easy to Use:** Simple, template-based injection framework to replace modules with optimized variants.

**Updates:**

*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
*   **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
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

<h2 id="show-cases">🌟 Show Cases</h2>

<div>
    <h3>Unleash the Power of Local LLMs</h3>
</div>

*   **GPT-4/o1-level Local VSCode Copilot:** Experience a powerful coding assistant on a desktop with only 24GB VRAM.

    <p align="center">
        <img alt="VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=70%>
    </p>

*   **Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, attaining performance that surpasses GPT-4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=70%>
    </p>

    *   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading, incorporating advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Provides an OpenAI and Ollama compatible API, making it easy to integrate as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends.

    <p align="center">
        <img alt="VSCode Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=70%>
    </p>

<!--
    <h3>1M Context Local Inference on a Desktop with Only 24GB VRAM</h3>
    <p align="center">

    https://github.com/user-attachments/assets/a865e5e4-bca3-401e-94b8-af3c080e6c12

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

<h2 id="quick-start">🚀 Quick Start</h2>

Getting started with KTransformers is simple! Follow the steps below to set up and start using it.

we have already supported vendors:

- Metax
- Sanechips (ZhuFeng V1.0)
- Intel
- Ascend
- Kunpeng
- AMD

### 📥 Installation

To install KTransformers, follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

<h2 id="tutorial">📃 Brief Injection Tutorial</h2>

KTransformers utilizes a user-friendly, template-based injection framework. This framework enables researchers to easily substitute original torch modules with optimized variants and effortlessly combine diverse optimizations, enabling exploration of their synergistic effects.

</br>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers concentrates on local deployments that are resource-constrained. We prioritize heterogeneous computing opportunities like GPU/CPU offloading of quantized models. For example, it supports the efficient <a href="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a href="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a href="doc/en/operators/llamafile.md">here</a>.

<h3>Example Usage</h3>
To utilize the provided kernels, users only need to create a YAML-based injection template and add the call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

In this example, the AutoModel is first initialized on the meta device to avoid occupying any memory resources. Then, `optimize_and_load_gguf` iterates through all sub-modules of the model, matches rules specified in your YAML rule file, and replaces them with advanced modules as specified.

After injection, the original `generate` interface is available, but we also provide a compatible `prefill_and_generate` method, which enables further optimizations like CUDAGraph to improve generation speed.

<h3>How to custom your model</h3>

A detailed tutorial of the injection and multi-GPU using DeepSeek-V2 as an example is given [here](doc/en/injection_tutorial.md).

Below is an example of a YAML template for replacing all original Linear modules with Marlin, an advanced 4-bit quantization kernel.

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

Each rule in the YAML file has two parts: `match` and `replace`. The `match` part specifies which module should be replaced, and the `replace` part specifies the module to be injected into the model along with the initialization keywords.

You can find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14, two SOTA MoE models, in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. These templates are used to power the `local_chat.py` demo.

If you are interested in our design principles and the implementation of the injection framework, please refer to the [design document](doc/en/deepseek-v2-injection.md).

<h2 id="ack">Acknowledgment and Contributors</h2>

KTransformers is built upon the flexible foundation of Transformers and incorporates advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.

Developed by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>, KTransformers welcomes contributions from the community.

<h2 id="discussion">💬 Discussion</h2>

Join the community!  Feel free to open an issue or join our [WeChat Group](WeChatGroup.png) for discussion.

<h2 id="FAQ">🙋 FAQ</h2>

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).