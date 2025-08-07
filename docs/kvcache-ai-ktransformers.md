<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Advanced Optimizations</h1>
  <p><em>Experience cutting-edge Large Language Model (LLM) inference optimizations with KTransformers, a flexible and extensible framework.</em></p>
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

*   **Flexible Framework:** Easily integrate advanced kernel optimizations into your Hugging Face Transformers pipelines.
*   **Extensible Design:**  Built with Python and designed for easy customization and experimentation with new inference techniques.
*   **Transformers-Compatible:**  Seamlessly integrates with existing Transformers workflows, supporting the same interfaces.
*   **RESTful APIs:** Compatible with OpenAI and Ollama API standards for easy integration with various applications.
*   **Optimized Kernels:** Leverage cutting-edge kernels like Llamafile and Marlin for significant speedups.
*   **Multi-Platform Support:** Works with various hardware vendors, including: Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.
*   **Rapid Updates:** Stay up-to-date with the latest optimizations and model support with frequent updates.

## Introduction

KTransformers, (Quick Transformers), provides a flexible, Python-centric framework for enhancing your 🤗 [Transformers](https://github.com/huggingface/transformers) experience. It unlocks advanced kernel optimizations and efficient placement/parallelism strategies to dramatically improve LLM inference performance.  With KTransformers, you can easily inject optimized modules with a single line of code, gaining access to a Transformers-compatible interface, compliant RESTful APIs (OpenAI and Ollama), and a simplified web UI for ChatGPT-like interactions. Our vision is to be the go-to platform for experimenting with innovative LLM inference optimizations, especially for local deployments.

## Updates

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
*   **July 11, 2025:** Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025:** Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025:** Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025:** Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))

https://github.com/user-attachments/assets/fafe8aec-4e22-49a8-8553-59fb5c6b00a2

*   **Apr 9, 2025:** Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
*   **Apr 2, 2025:** Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).

https://github.com/user-attachments/assets/faa3bda2-928b-45a7-b44f-21e12ec84b8a

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

## Show Cases

<div>
<h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

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
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version using only 21GB VRAM and 136GB DRAM on a local desktop, which scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

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

**More advanced features will be coming soon, so stay tuned!**

## Quick Start

Getting started with KTransformers is simple! Follow the steps below to set up and start using it.

We have already supported vendors:

*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

### Installation

To install KTransformers, follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## Brief Injection Tutorial

KTransformers uses a user-friendly, template-based injection framework. This allows researchers to easily replace original torch modules with optimized variants. It also simplifies combining multiple optimizations, allowing the exploration of synergistic effects.

<br/>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

Given that vLLM already serves as a great framework for large-scale deployment optimizations, KTransformers is particularly focused on local deployments that are constrained by limited resources. We pay special attention to heterogeneous computing opportunities, such as GPU/CPU offloading of quantized models. For example, we support the efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU, respectively. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

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

### How to Customize Your Model

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

## Acknowledgment and Contributors

KTransformers benefits from the flexibility of Transformers and advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute back to the community by upstreaming our modifications.

KTransformers is actively maintained and developed by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/). We welcome new contributors.

## Discussion

If you have any questions, feel free to open an issue. Alternatively, join our WeChat group using the QR Code: [WeChat Group](WeChatGroup.png)

## FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).