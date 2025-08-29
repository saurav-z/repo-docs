<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge LLM Inference with Optimized Kernels</h1>
  <p><em>Accelerate your Large Language Model inference with KTransformers, a flexible framework designed for cutting-edge optimization.</em></p>
  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#FAQ">🙋 FAQ</a>
  </p>
</div>

## Key Features of KTransformers

*   🚀 **Enhanced Performance:** Experience significant speedups in LLM inference, especially for local deployments, leveraging optimized kernels.
*   🛠️ **Flexible Framework:**  Easily inject optimized modules with a single line of code, built for extensibility and experimentation.
*   ⚙️ **Transformers-Compatible:** Seamlessly integrates with the Hugging Face Transformers library.
*   🌐 **RESTful API Support:**  Compatible with OpenAI and Ollama APIs, simplifying integration with various applications.
*   💻 **Local Deployment Focused:** Optimized for resource-constrained environments, maximizing performance on local machines.
*   📦 **Comprehensive Hardware Support:** Supports a wide range of hardware vendors, including Intel, AMD, and more.
*   📝 **Detailed Documentation & Tutorials:** Includes a comprehensive [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) and [FAQ](doc/en/FAQ.md).

## What is KTransformers?

KTransformers (pronounced "Quick Transformers") is a Python-centric framework designed to optimize and accelerate the inference of Large Language Models (LLMs). It offers a flexible platform for researchers and developers to experiment with advanced kernel optimizations and placement/parallelism strategies, achieving significant speedups, especially in resource-constrained environments. **[Learn more on the original repository](https://github.com/kvcache-ai/ktransformers).**

## 🔥 Recent Updates

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

## 🌟 Show Cases: Real-World Performance Boosts

KTransformers delivers impressive performance improvements, as demonstrated in the following use cases:

<div>
<h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:**  Running its Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Prefill Speed (tokens/s):**
        *   KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
        *   Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
    *   **Decode Speed (tokens/s):**
        *   KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
        *   Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
    *   **Upcoming Open Source Release:**
        *   AMX optimizations and selective expert activation will be open-sourced in V0.3.
        *   Currently available only in preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).
*   **Local 236B DeepSeek-Coder-V2:** Running its Q4_K_M version using only 21GB VRAM and 136GB DRAM, attainable on a local desktop machine, which scores even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

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

<!-- <h3>1M Context Local Inference on a Desktop with Only 24GB VRAM</h3>
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

**Stay tuned for more advanced features coming soon!**

## 🚀 Quick Start: Get Up and Running

KTransformers is designed for ease of use. Get started quickly by following these steps:

### 📥 Installation

For detailed installation instructions, please consult the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

### Supported Vendors

*   Metax
*   Sanechips (ZhuFeng V1.0)
*   Intel
*   Ascend
*   Kunpeng
*   AMD

## 📃 Brief Injection Tutorial:  Customize and Optimize

KTransformers employs a user-friendly, template-based injection framework that allows you to easily replace original PyTorch modules with optimized versions, enabling customized optimizations.

</br>
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers is particularly focused on local deployments with limited resources, supporting heterogeneous computing opportunities. It supports efficient kernels like [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin). More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

Utilize the provided kernels by creating a YAML-based injection template and adding a call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

In this example, `AutoModel` is initialized on the meta device. `optimize_and_load_gguf` iterates through the model's sub-modules, replaces them based on your YAML rules, and injects the optimized modules. After injection, the original `generate` interface is available, along with an optimized `prefill_and_generate` method for enhanced speed.

### How to Custom Your Model

A detailed tutorial on injection and multi-GPU usage with DeepSeek-V2 can be found [here](doc/en/injection_tutorial.md).

Here's an example YAML template:

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

Each YAML rule contains `match` (specifying modules to replace) and `replace` (specifying the injected module and initialization parameters).

Example rule templates for DeepSeek-V2 and Qwen2-57B-A14 are located in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory, used in the `local_chat.py` demo. For detailed design principles, refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🙋 FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).

## ✨ Acknowledgment & Collaboration

KTransformers is built upon the solid foundation of the Hugging Face Transformers library, with contributions from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. The KTransformers project is maintained and developed by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/). We encourage community contributions to improve KTransformers.

## 🤝 Discussion & Support

For any questions or discussions, please open an issue. You are also welcome to join our WeChat group (QR Code below):

<p align="center">
   <img src="WeChatGroup.png" alt="WeChat Group QR Code" width="200">
</p>