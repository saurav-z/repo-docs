<div align="center">
  <picture>
      <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h1>KTransformers: Supercharge Your LLM Inference</h1>
  <p><em>Experience blazing-fast inference for Large Language Models with KTransformers, a flexible framework for optimized kernel integration.</em></p>

  <p>
    <a href="#features">🌟 Key Features</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#show-cases">✨ Show Cases</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussions</a> |
    <a href="#faq">🙋 FAQ</a>
  </p>
</div>

---

## Overview

KTransformers is a powerful, Python-centric framework designed to accelerate inference for Hugging Face Transformers models.  By integrating advanced kernel optimizations and leveraging intelligent placement and parallelism strategies, KTransformers unlocks significant speedups and resource efficiency, especially for local deployments. [Visit the original repository](https://github.com/kvcache-ai/ktransformers).

## <a id="features">🌟 Key Features</a>

*   **Flexible Framework:** Easily integrate optimized modules with a single line of code.
*   **Transformers Compatibility:**  Seamlessly works with the Hugging Face Transformers ecosystem.
*   **OpenAI & Ollama API Compliance:** Provides RESTful APIs compatible with OpenAI and Ollama.
*   **Simplified Web UI:** Offers a user-friendly, ChatGPT-like web UI.
*   **Heterogeneous Computing Support:** Optimizes for GPU/CPU offloading of quantized models.
*   **Cutting-Edge Kernel Integration:** Supports and integrates with kernels like GGUF/GGML, Llamafile, Marlin, and flashinfer.
*   **Active Development:** Continuously updated with support for new models, features, and optimizations.

---

## <a id="updates">🔥 Recent Updates</a>

*   **July 26, 2025:** Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
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

---

## <a id="show-cases">✨ Show Cases</a>

KTransformers delivers impressive performance gains, enabling powerful LLM experiences on resource-constrained hardware.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:**  Run Q4_K_M version using only 14GB VRAM and 382GB DRAM([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Significant Speedups:** Achieve up to **27.79x speedup** in prefill and **3.03x speedup** in decode compared to llama.cpp on dual-socket CPUs.
    *   **AMX Optimization:** Optimized AMX-based MoE kernels will be open-sourced in V0.3.

*   **Local 236B DeepSeek-Coder-V2:** Run the Q4_K_M version using only 21GB VRAM and 136GB DRAM, surpassing GPT-4-0613 in the [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

    <p align="center">
      <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
    </p>

    *   **Faster Performance:** 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Easily integrate with Tabby and other frontends via OpenAI and Ollama compatibility.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" alt="VSCode Copilot Demo"  />
    </p>

---

## <a id="quick-start">🚀 Quick Start</a>

Getting started with KTransformers is straightforward!

### 📥 Installation

Install KTransformers by following the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

---

## <a id="tutorial">📃 Brief Injection Tutorial</a>

KTransformers uses a template-based injection framework, enabling researchers to easily swap original torch modules with optimized variants. This simplifies experimentation with multiple optimizations and their synergistic effects.

<p align="center">
  <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</p>

KTransformers focuses on local deployments with limited resources, with special attention to heterogeneous computing opportunities. For example, it supports the efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

Use the provided kernels by creating a YAML-based injection template and calling `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

This example initializes the AutoModel on the meta device, avoiding memory allocation.  `optimize_and_load_gguf` then replaces modules with advanced alternatives according to your YAML rules.  The original `generate` interface remains available, with a compatible `prefill_and_generate` method offering further optimizations.

### Customizing Your Model

A detailed tutorial for injection and multi-GPU usage with DeepSeek-V2 as an example is available [here](doc/en/injection_tutorial.md).

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

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. These templates are used in the `local_chat.py` demo.

For a deeper understanding of the injection framework's design principles and implementation, please refer to the [design document](doc/en/deepseek-v2-injection.md).

---

## <a id="ack">Acknowledgment and Contributors</a>

KTransformers leverages the Transformers framework and benefits from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang and flashinfer.  We plan to contribute modifications back to the community.

KTransformers is actively maintained by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>.  New contributions are welcome.

---

## <a id="ack">Discussion</a>

For questions, please open an issue or join our WeChat group. [WeChat Group](WeChatGroup.png)

---

## <a id="FAQ">🙋 FAQ</a>

Find answers to common questions in the [FAQ](doc/en/FAQ.md).