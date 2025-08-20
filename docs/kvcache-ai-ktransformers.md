<div align="center">
  <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  <h1>KTransformers: Supercharge Your LLM Inference with Kernel Optimizations</h1>
  <p><b>Unlock the power of optimized LLM inference with KTransformers, a flexible framework for blazing-fast performance.</b></p>

  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers"> 🔗 View on GitHub</a>
  </p>
</div>

## Key Features

*   **Accelerated Inference:** Experience significant speedups in LLM inference through advanced kernel optimizations.
*   **Flexible Framework:**  Easily integrate and experiment with cutting-edge inference techniques.
*   **Transformers Compatibility:** Seamlessly integrates with Hugging Face Transformers for a familiar user experience.
*   **Optimized Modules:**  Inject optimized modules with a single line of code for performance gains.
*   **OpenAI & Ollama API Compliance:** Supports RESTful APIs compliant with OpenAI and Ollama for easy integration.
*   **Simplified Web UI:** Includes a streamlined, ChatGPT-like web UI for quick testing and experimentation.
*   **Hardware Acceleration:** Supports a variety of hardware, including AMD, Intel, and Ascend GPUs.

## What's New

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

KTransformers enables you to run powerful LLMs locally and efficiently.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Performance Boost:** Achieves up to 27.79x speedup in prefill and 3.03x in decode compared to llama.cpp.
    *   **Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Run its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, scoring even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).
    *   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Integrates seamlessly as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

<p align="center">

https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c

</p>

**Stay tuned for more advanced features coming soon!**

## Quick Start

Ready to get started?

### Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

## Injection Tutorial

KTransformers uses a template-based injection framework to make it easy to swap out original torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments and heterogeneous computing, supporting efficient kernels like <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a>.

### Example Usage

Use a YAML-based injection template and call `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

### Customizing Your Model

A detailed tutorial, using DeepSeek-V2 as an example, is available [here](doc/en/injection_tutorial.md).

Example YAML template:

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

Find example rule templates in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For more on design principles and implementation, see the [design document](doc/en/deepseek-v2-injection.md).

## Acknowledgment and Contributors

KTransformers is built on the foundation of Transformers, and benefits from projects like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer.

KTransformers is actively maintained by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## Discussion

Have questions or want to connect?  Open an issue or join our WeChat group: [WeChat Group](WeChatGroup.png)

## FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).