<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Optimized Kernels</h1>
  <p><b>KTransformers empowers you to unlock blazing-fast Large Language Model (LLM) inference on your hardware, offering a flexible framework for advanced kernel optimizations.</b></p>
  <p>
    <a href="#show-cases">🌟 Show Cases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers"> <b>View on GitHub</b></a>
  </p>
</div>

## Key Features

*   ⚡ **Blazing Fast Inference:** Achieve significant speedups in LLM inference through optimized kernels and placement/parallelism strategies.
*   🛠️ **Flexible & Extensible:** Easily integrate and experiment with advanced optimizations using a Python-centric framework.
*   🧠 **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers ecosystem.
*   💡 **OpenAI & Ollama API Support:** Provides compatible interfaces for easy integration with existing tools and applications.
*   💻 **Local Deployment Focus:** Optimized for resource-constrained environments, enabling powerful LLM capabilities on your desktop.
*   ⚙️ **Heterogeneous Computing:** Leverages GPU/CPU offloading and quantization techniques for maximum performance.

## What's New

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

## Show Cases

KTransformers empowers you to run state-of-the-art LLMs locally with impressive speedups, even on resource-constrained hardware.

<div>
<h3>GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>
</div>

### Key Achievements

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4\_K\_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Significant Speedups:**
        *   **Prefill Speed:** Up to **27.79x** faster than llama.cpp (255.26 tokens/s with optimized AMX-based MoE kernel).
        *   **Decode Speed:** Up to **3.03x** faster than llama.cpp (13.69 tokens/s).
    *   **Upcoming Open Source Release:** AMX optimizations and selective expert activation will be open-sourced in V0.3.
*   **Local 236B DeepSeek-Coder-V2:** Achieved state-of-the-art performance running its Q4_K_M version using only 21GB VRAM and 136GB DRAM, exceeding GPT-4-0613 in BigCodeBench.
    *   **Faster Inference:** 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading using advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Seamlessly integrated as a backend for [Tabby](https://github.com/TabbyML/tabby) and other frontends via OpenAI and Ollama compatibility.

<!--
### 1M Context with Local Inference
*   **1M Context InternLM 2.5 7B**: Operates at full bf16 precision, utilizing 24GB VRAM and 150GB DRAM, which is feasible on a local desktop setup. It achieves a 92.88% success rate on the 1M "Needle In a Haystack" test and 100% on the 128K NIAH test.
*   **Enhanced Speed**: Reaches 16.91 tokens/s for generation with a 1M context using sparse attention, powered by llamafile kernels. This method is over 10 times faster than full attention approach of llama.cpp.
*   **Flexible Sparse Attention Framework**: Offers a flexible block sparse attention framework for CPU offloaded decoding. Compatible with SnapKV, Quest, and InfLLm. Further information is available [here](./doc/en/long_context_introduction.md).
 -->

**Stay tuned for more advanced features coming soon!**

## Quick Start

Get started with KTransformers in minutes!

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to set up your environment.

## Brief Injection Tutorial

KTransformers utilizes a template-based injection framework, enabling easy integration of optimized modules.  This simplifies combining multiple optimizations for experimentation.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments with resource constraints, using heterogeneous computing (GPU/CPU offloading) with Llamafile and Marlin kernels. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

Use YAML-based injection templates and the `optimize_and_load_gguf` function.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

This example initializes the AutoModel, then `optimize_and_load_gguf` replaces modules based on your YAML rules.

### Customizing Your Model

Create YAML templates to specify module replacements.  A detailed tutorial for DeepSeek-V2 injection is [here](doc/en/injection_tutorial.md).

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

The `match` section specifies modules to replace, and `replace` specifies the replacement module and initialization parameters.

Example rule templates for DeepSeek-V2 and Qwen2-57B-A14 are in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. See the [design document](doc/en/deepseek-v2-injection.md) for design principles.

## Acknowledgment and Contributors

KTransformers is built on the foundation of the Hugging Face Transformers library, along with leveraging advanced kernels such as GGUF/GGML, Llamafile, Marlin, sglang and flashinfer.  We plan to contribute our modifications back to the community.

Developed by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  We welcome new contributors.

## Discussion

Join the community!

*   Open an issue with any questions.
*   Discuss with us in the [WeChat Group](WeChatGroup.png).

## FAQ

Find answers to frequently asked questions in the [FAQ](doc/en/FAQ.md).