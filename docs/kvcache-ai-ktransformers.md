<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h3>Supercharge Your LLM Inference: KTransformers Unleashes Cutting-Edge Optimizations for Faster, More Efficient Performance</h3>
  <p>
    <a href="#features">🚀 Key Features</a> |
    <a href="#showcases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">🔗  View on GitHub</a>
  </p>
</div>

##  🚀 Key Features
KTransformers is a flexible, Python-centric framework built for optimizing your Hugging Face Transformers experience.  Here's what you can do with it:

*   **Advanced Kernel Optimizations:** Experience significant speedups through optimized kernels.
*   **Flexible Architecture:** Easily integrate optimized modules with a single line of code.
*   **Transformers Compatibility:**  Works seamlessly with the Hugging Face Transformers interface.
*   **API Support:**  Includes RESTful APIs compatible with OpenAI and Ollama.
*   **Simplified UI:**  Includes a user-friendly, ChatGPT-like web UI.
*   **Heterogeneous Computing:** Supports GPU/CPU offloading and quantization.
*   **Model Compatibility:** Support of many models, including SmallThinker, GLM4-MoE, Kimi-K2, Llama 4, and more.
*   **Hardware Support:** Supports Intel Arc GPU, AMD GPU (ROCm), and AMX-Int8/BF16.

## 🎉 Introduction
KTransformers, pronounced "Quick Transformers," is designed to enhance your 🤗 Transformers experience with advanced kernel optimizations and placement/parallelism strategies.  It offers a flexible platform for experimenting with innovative LLM inference optimizations.

## 🔥 Updates
Stay up-to-date with the latest features and improvements:

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2 ([Tutorial](./doc/en/Kimi-K2.md))
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

## 🌟 Show Cases

KTransformers delivers impressive performance gains on local hardware.

<div>
    <h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

<p align="center">
    <img alt="VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=70%>
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   Prefill Speed (tokens/s):
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   Decode Speed (tokens/s):
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   Upcoming Open Source Release:
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM on a local desktop machine, which scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Integrate seamlessly as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via OpenAI and Ollama compatible API.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=70%>
</p>

## 🚀 Quick Start

Follow these steps to get started with KTransformers.

Supported Vendors:
*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

### 📥 Installation

Refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Brief Injection Tutorial

KTransformers uses a template-based injection framework for easy module replacement.  This allows for experimenting with synergistic optimizations.

<br/>
<p align="center">
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</p>

KTransformers focuses on local deployments, especially for resource-constrained environments. It supports heterogeneous computing, including GPU/CPU offloading and quantization. For example, we support the efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU, respectively. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage
Users can use the provided kernels by creating a YAML-based injection template and adding the call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

In this example, the AutoModel is first initialized on the meta device to avoid occupying any memory resources. Then, `optimize_and_load_gguf` iterates through all sub-modules of the model, matches rules specified in your YAML rule file, and replaces them with advanced modules as specified.

After injection, the original `generate` interface is available, but we also provide a compatible `prefill_and_generate` method, which enables further optimizations like CUDAGraph to improve generation speed.

### How to customize your model
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

For details on the injection framework's design and implementation, see the [design document](doc/en/deepseek-v2-injection.md).

## <h2 id="ack">Acknowledgment and Contributors</h2>

KTransformers builds on the Transformers framework and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  We plan to contribute back to the community.

KTransformers is actively maintained by the MADSys group at Tsinghua University and members from Approaching.AI. We welcome new contributors.

## <h2 id="ack">Discussion</h2>
For questions, open an issue or join our WeChat group. QR Code: [WeChat Group](WeChatGroup.png)

## <h2 id="FAQ">🙋 FAQ</h2>
Find answers to common questions in the [FAQ](doc/en/FAQ.md).