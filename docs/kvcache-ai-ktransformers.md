<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
</div>

# KTransformers: Supercharge Your LLM Inference with Kernel Optimizations

**KTransformers empowers you to optimize and accelerate Large Language Model (LLM) inference, enabling faster and more efficient deployments.** [View the original repository](https://github.com/kvcache-ai/ktransformers)

**Key Features:**

*   🚀 **Accelerated Inference:** Boost LLM performance with advanced kernel optimizations.
*   🧩 **Flexible & Extensible:**  Easily integrate optimized modules and experiment with cutting-edge inference techniques.
*   🛠️ **Transformers Compatibility:** Seamlessly works with the Hugging Face Transformers library.
*   🌐 **API Support:**  Includes RESTful APIs compatible with OpenAI and Ollama.
*   💻 **Local Deployment Focused:** Optimized for resource-constrained local deployments with GPU/CPU offloading.
*   🤝 **Community Driven:**  Active development with contributions from Tsinghua University and Approaching.AI.

##  🎉 Introduction
KTransformers, short for "Quick Transformers", revolutionizes your 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers) experience by incorporating advanced kernel optimizations and strategic placement/parallelism techniques, allowing for a more efficient and faster LLM inference. Designed with flexibility at its core, KTransformers allows users to easily inject optimized modules for a Transformers-compatible interface, OpenAI and Ollama compliant APIs, and simplified web UI.

##  🔥 Updates
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

<!-- * **Aug 28, 2024**: Support 1M context under the InternLM2.5-7B-Chat-1M model, utilizing 24GB of VRAM and 150GB of DRAM. The detailed tutorial is [here](./doc/en/long_context_tutorial.md). -->

##  🌟 Show Cases
### Achieve GPT-4/o1-Level Performance Locally
Here are some key examples of how KTransformers can enhance your LLM deployments:
*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Significant Speedups:** Achieve substantial speed improvements compared to baseline implementations:
        *   **Prefill Speed:** Up to 27.79x faster using AMX-based optimizations (V0.3).
        *   **Decode Speed:** Up to 3.03x faster.
    *   **Upcoming Open Source Release:** AMX optimizations and expert activation will be open-sourced in V0.3.  Preview binary is available [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Run Q4\_K\_M version using only 21GB VRAM and 136GB DRAM, exceeding GPT4-0613 in the [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).
    *   **Faster Performance:** Generate at 13.6 tokens/s with 2K prompt prefill.
    *   **VSCode Integration:** Compatible with OpenAI and Ollama APIs for seamless integration with tools like [Tabby](https://github.com/TabbyML/tabby).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

<p align="center">

https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c

</p>

<!--
### 1M Context Local Inference

*   **1M Context InternLM 2.5 7B**: Achieves high success rates on the "Needle In a Haystack" test using only 24GB VRAM and 150GB DRAM.

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
*   **Enhanced Speed:**  Generates with a 1M context using sparse attention, powered by llamafile kernels, over 10 times faster than llama.cpp.

*   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
 -->

**More advanced features are coming soon!**

## 🚀 Quick Start
KTransformers is easy to use! Follow the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started.
KTransformers already supports:
*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

## 📃 Brief Injection Tutorial
KTransformers offers a user-friendly injection framework, enabling researchers to replace original torch modules with optimized variants.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers emphasizes local deployments, focusing on GPU/CPU offloading of quantized models. We support kernels like [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).

### Example Usage
Create a YAML-based injection template and use `optimize_and_load_gguf`:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

After injection, use the original `generate` interface, or utilize `prefill_and_generate` for further optimizations.

### Customizing Your Model
Find detailed tutorials on injection and multi-GPU configuration, using DeepSeek-V2 as an example [here](doc/en/injection_tutorial.md).

YAML template example to replace Linear modules:

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

For design principles and the injection framework implementation, see the [design document](doc/en/deepseek-v2-injection.md).

##  🤝 Acknowledgment and Contributors
KTransformers is built upon the foundation of Transformers, leveraging kernels like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer. We are planning to contribute back to the community.  KTransformers is actively maintained and developed by the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/).

##  💬 Discussion

For questions, open an issue. Join our [WeChat Group](WeChatGroup.png) for discussion.

## 🙋 FAQ
Find answers to common questions in the [FAQ](doc/en/FAQ.md).