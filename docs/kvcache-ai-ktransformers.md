<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h3>Supercharge your Hugging Face Transformers experience with KTransformers: experience cutting-edge LLM inference optimizations!</h3>
  <strong><a href="#key-features">🚀 Key Features</a> | <a href="#showcases">🌟 Showcases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a>|<a href="#faq"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">💾  GitHub</a></strong>
</div>

## What is KTransformers?

KTransformers is a Python-centric framework designed to provide flexible and extensible solutions for LLM inference optimizations.  It enhances your Hugging Face Transformers experience with advanced kernel optimizations and placement/parallelism strategies.

## 🔥 Updates

*   **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
*   **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
*   **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
*   **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
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
<!-- * **Aug 28, 2024**: Support 1M context under the InternLM2.5-7B-Chat-1M model, utilizing 24GB of VRAM and 150GB of DRAM. The detailed tutorial is [here](./doc/en/long_context_tutorial.md). -->

## <a id="key-features"></a> 🚀 Key Features

*   **Flexible Framework:** Easily experiment with cutting-edge LLM inference optimizations through a Python-centric design.
*   **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   **Kernel Optimization:** Optimized modules are implemented with a single line of code to boost performance.
*   **RESTful APIs:** Provides compliant APIs with OpenAI and Ollama standards.
*   **Simplified UI:**  Includes a user-friendly, ChatGPT-like web UI.
*   **Heterogeneous Computing Support:**  Optimizes for heterogeneous computing environments, including GPU/CPU offloading.
*   **Model Support:** Supports multiple models.

## <a id="showcases"></a> 🌟 Showcases

*   **Local VSCode Copilot-Level Performance:** Experience GPT-4/o1-level Copilot capabilities on a desktop with only 24GB of VRAM.

<p align="center">

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed:**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Achieving up to **27.79× speedup** compared to llama.cpp.
    *   **Decode Speed:**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Achieving up to **3.03× speedup** compared to llama.cpp.
    *   **Upcoming Open Source Release:** AMX optimizations and selective expert activation in V0.3.
    *   Available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Runs its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, surpassing GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Faster Speed:** Achieves 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    *   **VSCode Integration:** Integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends.

    <p align="center">

    https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c

    </p>

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
    *   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.
    *   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
    -->
    **More advanced features will coming soon, so stay tuned!**

## <a id="quick-start"></a> 🚀 Quick Start

1.  **Installation:** Follow the instructions in the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

    Supported Vendors:
    *   Metax
    *   Sanechips (ZhuFeng V1.0)
    *   Intel
    *   Ascend
    *   Kunpeng
    *   AMD

## <a id="tutorial"></a> 📃 Injection Tutorial

KTransformers features a user-friendly, template-based injection framework for easy module optimization.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

Leveraging frameworks like vLLM, KTransformers focuses on local deployments with limited resources. It supports efficient kernels such as <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a>. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Use YAML templates to replace original torch modules with optimized variants.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

The `optimize_and_load_gguf` function iterates through sub-modules, matching rules specified in your YAML file and replacing them with advanced modules.

The `generate` interface is available. The `prefill_and_generate` method enables optimizations like CUDAGraph.

### Custom Model

A detailed tutorial of the injection and multi-GPU using DeepSeek-V2 as an example is given [here](doc/en/injection_tutorial.md).

YAML template example:

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

See [ktransformers/optimize/optimize\_rules](ktransformers/optimize/optimize_rules) for DeepSeek-V2 and Qwen2-57B-A14 templates.

Refer to the [design document](doc/en/deepseek-v2-injection.md) for design principles.

## Acknowledgment and Contributors

KTransformers builds upon Transformers and benefits from kernels such as GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.  Contributions are planned to be upstreamed to the community.

Maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.

## <a id="discussion"></a> Discussion

If you have any questions, open an issue. Join the WeChat group for discussion: [WeChat Group](WeChatGroup.png)

## <a id="faq"></a> 🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).