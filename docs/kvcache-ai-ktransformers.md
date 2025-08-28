<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Unleash LLM Performance with Optimized Inference</h1>
  <p><em>Supercharge your Hugging Face Transformers with advanced kernel optimizations and flexible deployment strategies for lightning-fast LLM inference.</em></p>
  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">🔗  Original Repo</a>
  </p>
</div>

## Key Features

*   **Optimized Kernels:** Accelerate inference with cutting-edge kernel optimizations, including support for various hardware platforms and quantization techniques.
*   **Flexible Injection Framework:** Easily inject optimized modules into your existing Transformers models with a simple, Python-centric framework.
*   **OpenAI & Ollama Compatibility:** Seamlessly integrate with OpenAI and Ollama APIs, enabling you to use KTransformers with popular frontends.
*   **Multi-GPU Support:** Distribute model inference across multiple GPUs for increased throughput.
*   **Comprehensive Hardware Support:** Compatibility with major vendors including Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.

## 🔥 Updates
*   **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
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

**KTransformers empowers you to run state-of-the-art LLMs locally, with impressive performance gains.**

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4_K_M version using just 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Speedups:**
        *   Prefill Speed (tokens/s):
            *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
            *   Up to **27.79x faster** than llama.cpp with 2×32 cores (10.31 tokens/s).
        *   Decode Speed (tokens/s):
            *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
            *   Up to **3.03x faster** than llama.cpp with 2×32 cores (4.51 tokens/s).
    *   **Coming Soon:** AMX optimizations and selective expert activation will be open-sourced in V0.3. (Preview binary distribution available [here](./doc/en/DeepseekR1_V3_tutorial.md)).

*   **Local 236B DeepSeek-Coder-V2:** Run its Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, outperforming GPT-4 0613 on [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <picture>
        <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
      </picture>
    </p>

    *   **Performance:** 126 tokens/s (2K prompt prefill), 13.6 tokens/s (generation).
    *   **Integration:**  Compatible with OpenAI and Ollama APIs, for use with Tabby and other frontends.

    <p align="center">
      <picture>
        <img alt="Copilot Demo" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=60%>
      </picture>
    </p>

    <!--
    ### 1M Context Local Inference on a Desktop with Only 24GB VRAM
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

## 🚀 Quick Start

Follow the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started.

## 📃 Tutorial: Injecting Optimized Modules

KTransformers offers a user-friendly, template-based injection framework for seamlessly integrating optimized modules. This enables easy experimentation with various optimization techniques.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on resource-constrained local deployments. It leverages heterogeneous computing with GPU/CPU offloading of quantized models. For example, KTransformers supports Llamafile and Marlin kernels for CPU and GPU, respectively. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

To utilize the provided kernels, create a YAML-based injection template and add the call to `optimize_and_load_gguf` before using the Transformers model:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

The `optimize_and_load_gguf` function iterates through all model sub-modules and replaces them with optimized modules according to the rules specified in your YAML file.
The original `generate` interface is available, and we provide a compatible `prefill_and_generate` method for further optimizations.

### Customizing Your Model

A detailed tutorial of the injection is given [here](doc/en/injection_tutorial.md).

YAML template example for replacing Linear modules with Marlin:

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

For design details of the injection framework, see the [design document](doc/en/deepseek-v2-injection.md).

## ❤️ Acknowledgments

KTransformers builds upon the Hugging Face Transformers library and benefits from contributions from projects like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.

Developed by the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/).

## 🙋 FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).