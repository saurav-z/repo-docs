<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

# KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations

**KTransformers empowers you to run large language models faster and more efficiently by integrating advanced kernel optimizations and placement strategies, making LLM inference on your local machine a reality.**

**[🌟 Show Cases](#show-cases) | [🚀 Quick Start](#quick-start) | [📃 Tutorial](#tutorial) | [💬 Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](#FAQ) | [🌐 Original Repo](https://github.com/kvcache-ai/ktransformers)**

## Key Features:

*   **Enhanced Speed and Efficiency:** Experience significant speedups in LLM inference using optimized kernels and strategies.
*   **Flexible Framework:**  A Python-centric design built for extensibility, allowing easy integration of optimized modules.
*   **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   **OpenAI/Ollama API Compliance:**  Supports RESTful APIs compatible with OpenAI and Ollama, enabling integration with various frontends.
*   **Optimized for Local Deployment:** Focuses on optimizing inference for resource-constrained local environments, maximizing performance on limited VRAM.
*   **Multi-Platform Support**: Support multiple vendors, including Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.
*   **Community-Driven:** Actively developed by the MADSys group at Tsinghua University and members from Approaching.AI.

## 🔥 Updates

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

## 🌟 Show Cases

### Achieve GPT-4/o1-level Local Copilot Performance on a Desktop with Limited VRAM

#### Key Achievements:

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Significant Speedups:**
        *   Prefill Speed (tokens/s): Up to 286.55 (optimized AMX-based MoE kernel, V0.3 only, selectively using 6 experts)
        *   Decode Speed (tokens/s): Up to 13.69 (selectively using 6 experts, V0.3 only)
        *   Achieving up to **27.79x speedup** compared to llama.cpp (dual-socket, 2×32 cores) on prefill, and up to **3.03x speedup** on decode.
    *   **Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3 (Preview binary distribution available [here](./doc/en/DeepseekR1_V3_tutorial.md)).

*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version with only 21GB VRAM and 136GB DRAM, scoring better than GPT-4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

*   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation using MoE offloading and advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
*   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via an OpenAI and Ollama compatible API.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" alt="VSCode Copilot Integration" width=65%>
</p>

<!--
### 1M Context Local Inference on a Desktop with Only 24GB VRAM
*   **1M Context InternLM 2.5 7B**: Utilizes 24GB VRAM and 150GB DRAM, achieving a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.

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
-->

**More advanced features are coming soon!**

## 🚀 Quick Start

### 📥 Installation

1.  Follow the detailed [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## 📃 Brief Injection Tutorial

KTransformers provides a template-based injection framework to easily replace original PyTorch modules with optimized versions, enabling rapid experimentation with different optimizations and their combinations.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

This framework focuses on local deployments and heterogeneous computing, supporting efficient kernels like [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).

### Example Usage

Use YAML-based injection templates and the `optimize_and_load_gguf` function to integrate optimized kernels:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

`optimize_and_load_gguf` replaces model sub-modules based on your YAML rules.  The original `generate` interface is available, with an optimized `prefill_and_generate` method for enhanced speed.

### Customizing Your Model

Use YAML templates to specify module replacements. For instance, replace all `torch.nn.Linear` modules with `KTransformerLinear`:

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

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.  Refer to the [design document](doc/en/deepseek-v2-injection.md) for more details.

## <a id="ack"></a>Acknowledgment and Contributors

KTransformers leverages the flexibility of Transformers and benefits from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  We are planning to upstream our modifications to contribute back to the community.

KTransformers is developed by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## <a id="discussion"></a>Discussion

Open an issue or join our [WeChat Group](WeChatGroup.png) for discussions.

## <a id="FAQ"></a>🙋 FAQ

Check the [FAQ](doc/en/FAQ.md) for answers to common questions.