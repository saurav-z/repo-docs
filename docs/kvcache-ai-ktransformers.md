<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h1>KTransformers: Supercharge Your LLM Inference with Optimized Kernels</h1>
  <p><em>Experience lightning-fast and efficient LLM inference on your local hardware with KTransformers, a framework built for cutting-edge optimizations.</em></p>

  <p>
    <a href="#features">✨ Key Features</a> |
    <a href="#showcases">🌟 Showcases</a> |
    <a href="#quick-start">🚀 Quick Start</a> |
    <a href="#tutorial">📃 Tutorial</a> |
    <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬 Discussion</a> |
    <a href="#faq">🙋 FAQ</a> |
    <a href="https://github.com/kvcache-ai/ktransformers">GitHub Repo</a>
  </p>
</div>

## <a id="features">✨ Key Features</a>

KTransformers empowers developers to optimize their 🤗 Transformers experiences with advanced kernel optimizations and placement/parallelism strategies.

*   **Flexible Framework:** Python-centric design for easy extensibility and experimentation.
*   **Simplified Integration:** Inject optimized modules with a single line of code.
*   **Transformers-Compatible Interface:** Seamless integration with existing Transformers workflows.
*   **OpenAI/Ollama API Compatibility:** Supports RESTful APIs compliant with OpenAI and Ollama.
*   **Simplified Web UI:**  Easy-to-use, ChatGPT-like web UI available.
*   **GPU/CPU Offloading**: Support for heterogeneous computing opportunities like GPU/CPU offloading of quantized models.
*   **Model Compatibility**: Supports various models including Mixtral, Deepseek, Qwen, Llama, and more.

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

## <a id="showcases">🌟 Showcases</a>

KTransformers delivers significant performance gains and enables powerful applications on resource-constrained hardware.

### Local LLM Powerhouse: Run Cutting-Edge Models on Your Desktop

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version with just 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).

    *   **Performance Boost:**
        *   **Prefill Speed (tokens/s):** Up to **27.79x** faster than llama.cpp (255.26 tokens/s with optimized AMX-based MoE kernel).
        *   **Decode Speed (tokens/s):** Up to **3.03x** faster than llama.cpp (13.69 tokens/s with selective expert activation).
    *   **Coming Soon:** AMX optimizations and selective expert activation will be open-sourced in V0.3.

*   **Local 236B DeepSeek-Coder-V2:** Run its Q4_K_M version with only 21GB VRAM and 136GB DRAM, achieving even better results than GPT-4 in BigCodeBench.

    <p align="center">
      <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
    </p>

    *   **Faster Speed:** Achieve 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
    *   **VSCode Integration:** Integrated as a backend for Tabby and other frontends via an OpenAI and Ollama compatible API.

    <p align="center">
        <img alt="VSCode Copilot" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c">
    </p>

*   **Previous Showcases**:
    *   GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

        <p align="center">
          <img alt="VSCode Copilot" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285">
        </p>

<!--
### 1M Context Local Inference

  *   **1M Context InternLM 2.5 7B**: Runs at full bf16 precision with 24GB VRAM and 150GB DRAM.
  *   **High Accuracy:** Achieves 92.88% on the 1M "Needle In a Haystack" test.
  *   **Enhanced Speed:** Reaches 16.91 tokens/s with a 1M context using sparse attention.

    <p align="center">
      <img alt="Single Needle Retrieval 128K" src="./doc/assets/needle_128K.png" width=100%>
    </p>

    <p align="center">
      <img alt="Single Needle Retrieval 1000K" src="./doc/assets/needle_1M.png" width=100%>
    </p>

  *   **Flexible Sparse Attention Framework:** Supports CPU offloaded decoding and is compatible with SnapKV, Quest, and InfLLm.
-->
## <a id="quick-start">🚀 Quick Start</a>

Ready to experience the power of KTransformers? Get started with these simple steps:

### 📥 Installation

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) for a smooth setup.
*   **Supported Vendors**:
    *   Metax
    *   Sanechips (ZhuFeng V1.0)
    *   Intel
    *   Ascend
    *   Kunpeng
    *   AMD
    
## <a id="tutorial">📃 Brief Injection Tutorial</a>

KTransformers' core is a template-based injection framework for replacing original torch modules with optimized variants. This design allows you to easily combine multiple optimizations.

<p align="center">
  <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
</p>

KTransformers focuses on local deployments with limited resources, and offers GPU/CPU offloading options. It supports [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin) kernels.

### Example Usage

Use YAML-based injection templates and call `optimize_and_load_gguf` before using the Transformers model:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

Then, `optimize_and_load_gguf` replaces model sub-modules using rules in your YAML file.

### How to Customize Your Model

A detailed tutorial for injection, including a DeepSeek-V2 example, is [here](doc/en/injection_tutorial.md).

YAML example for replacing Linear modules with Marlin:

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

Rules have `match` (target module) and `replace` (optimized module). Find example rule templates for DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

For details on the injection framework, please see the [design document](doc/en/deepseek-v2-injection.md).

## <a id="ack">Acknowledgment and Contributors</a>

KTransformers is built upon the Transformer framework and benefits from kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We are actively contributing back to the community.

The project is maintained by the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. Join us to make KTransformers even better!

## <a id="discussion">Discussion</a>

Have questions?  Open an issue or join our WeChat group. QR Code: [WeChat Group](WeChatGroup.png)

## <a id="FAQ">🙋 FAQ</a>

Find answers to common questions in the [FAQ](doc/en/FAQ.md).

**[Go back to the top](#top)**