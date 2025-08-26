<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

# KTransformers: Accelerate LLM Inference with Optimized Kernels 🚀

**KTransformers empowers you to supercharge your Hugging Face Transformers experience with cutting-edge kernel optimizations for faster and more efficient LLM inference.**  Explore [the KTransformers GitHub repository](https://github.com/kvcache-ai/ktransformers) for the latest updates and to contribute.

*   [🌟 Show Cases](#show-cases) | [🚀 Quick Start](#quick-start) | [📃 Tutorial](#tutorial) | [💬 Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](#FAQ)

## Key Features

*   **Flexible Framework:** Easily inject optimized modules with a single line of code.
*   **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers library.
*   **OpenAI & Ollama API Support:** Includes compliant RESTful APIs for easy integration.
*   **Simplified Web UI:** Features a user-friendly, ChatGPT-like web UI.
*   **Heterogeneous Computing Support:** Leverages GPU/CPU offloading and quantization for optimal performance on resource-constrained systems.
*   **Cutting-Edge Optimizations:** Incorporates advanced kernels like Llamafile and Marlin for significant speedups.
*   **Multi-Platform Support:**  Support Windows, AMD, Intel, and more.

## What's New
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


## 🌟 Show Cases: Real-World Performance Boosts

*   **GPT-4/o1-level Local VSCode Copilot:** Experience near GPT-4 level performance on a local desktop with only 24GB VRAM.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Significant Speedups:**  Achieves up to **27.79x speedup** in prefill and up to **3.03x speedup** in decode compared to llama.cpp on a dual-socket, 2×32 core system.
    *   **AMX and Selective Expert Optimizations:** AMX optimizations and selective expert activation will be open-sourced in V0.3.
    *   **Preview Available:** Download the preview binary distribution [here](./doc/en/DeepseekR1_V3_tutorial.md).

*   **Local 236B DeepSeek-Coder-V2:**  Run the Q4_K_M version on a local desktop with 21GB VRAM and 136GB DRAM. It scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Faster Inference:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    *   **VSCode Integration:** Integrated as a backend for Tabby and other frontends through an OpenAI and Ollama compatible API.

    <p align="center">
        <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width="75%"/>
    </p>

## 🚀 Quick Start: Get Up and Running Fast

KTransformers is designed for easy setup.

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started.

## 📃 Brief Injection Tutorial: Customizing Your Models

KTransformers' injection framework allows you to easily swap original torch modules with optimized variants using YAML templates. This facilitates experimenting with different optimization techniques.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments.

### Example Usage

Use YAML templates and the `optimize_and_load_gguf` function:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

This example initializes the AutoModel, then `optimize_and_load_gguf` replaces modules with optimized versions based on your YAML rule file.

### Customizing Your Model

A detailed tutorial for injection with DeepSeek-V2 is [here](doc/en/injection_tutorial.md).

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

*   The `match` section specifies which modules to replace.
*   The `replace` section specifies the replacement module and its initialization keywords.

Find example templates for DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.  Read the [design document](doc/en/deepseek-v2-injection.md) for design principles.

## Acknowledgment and Contributors

KTransformers is built on the foundation of the Hugging Face Transformers library and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer.  We plan to contribute back to the community.

KTransformers is maintained by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/). We welcome new contributions!

## 💬 Discussion

Ask questions or discuss KTransformers by opening an issue or joining our WeChat group. QR Code: [WeChat Group](WeChatGroup.png)

## 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).