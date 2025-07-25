<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

# KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations

**KTransformers empowers you to run LLMs faster and more efficiently with advanced kernel optimizations.**

**[🌟 Show Cases](#show-cases) | [🚀 Quick Start](#quick-start) | [📃 Tutorial](#tutorial) | [💬 Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](#FAQ) | [📚 Documentation](https://kvcache-ai.github.io/ktransformers/) | [Original Repo](https://github.com/kvcache-ai/ktransformers)**

## Key Features

*   **Optimized Kernels:** Leverage cutting-edge optimizations for faster LLM inference.
*   **Flexible Framework:** Easily experiment with various optimization strategies.
*   **Transformers Compatibility:** Seamlessly integrate with Hugging Face Transformers.
*   **OpenAI & Ollama API Compatibility:** Deploy your models with standard API interfaces.
*   **Simplified Web UI:** Quickly deploy models with a user-friendly ChatGPT-like interface.
*   **Multi-Platform Support:** Supports a variety of vendors, including Metax, Sanechips, Intel, AMD, Ascend, and Kunpeng.

## 🔥 Recent Updates

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

## 🌟 Show Cases

### Local LLM Powerhouse: Experience GPT-4/o1-level performance on a desktop with limited resources.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version with only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version with only 21GB VRAM and 136GB DRAM, even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation via MoE offloading and optimized kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via an OpenAI and Ollama compatible API.

    <p align="center">

    https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c

    </p>

**More advanced features are coming soon!**

## 🚀 Quick Start

Get up and running with KTransformers quickly:

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to install KTransformers.

## 📃 Injection Tutorial

KTransformers simplifies optimization through a template-based injection framework, making it easy to integrate optimized modules with a single line of code.

<br/>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments with limited resources, paying special attention to heterogeneous computing opportunities. It supports the efficient [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for CPU and GPU, respectively. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

Create a YAML-based injection template and add the call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

After injection, the original `generate` interface is available, but we also provide a compatible `prefill_and_generate` method, which enables further optimizations like CUDAGraph to improve generation speed.

### Customizing Your Model

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

Example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 can be found in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. These templates are used to power the `local_chat.py` demo.

For design principles and the implementation of the injection framework, please refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🙋 FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).

## 🙏 Acknowledgment and Contributors

KTransformers is built upon the foundation of the Transformers framework, with contributions from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. The project is actively maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## 💬 Discussion

Have questions? Open an issue or join our WeChat group for discussion. [WeChat Group](WeChatGroup.png)