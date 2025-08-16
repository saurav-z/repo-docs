<div align="center">
  <picture>
    <img alt="KTransformers Logo" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>

  <h1>KTransformers: Supercharge Your LLM Inference with Cutting-Edge Optimizations</h1>
  <p><em>Accelerate your Hugging Face Transformers experience with flexible kernel optimizations and advanced deployment strategies.</em></p>
  <p>
    <strong><a href="#show-cases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a>|<a href="#FAQ"> 🙋 FAQ</a> | <a href="https://github.com/kvcache-ai/ktransformers">🔗 View on GitHub</a></strong>
  </p>
</div>

## Key Features

*   **Flexible Framework:** Easily integrate optimized modules with a single line of code.
*   **Transformers Compatibility:** Seamlessly works with the Hugging Face Transformers ecosystem.
*   **OpenAI/Ollama-Compliant APIs:** Supports standard API interfaces for easy integration.
*   **Simplified Web UI:** Includes a ChatGPT-like web UI for quick experimentation.
*   **Cutting-Edge Optimizations:** Explore and implement innovative LLM inference techniques.
*   **Support for Various Vendors:** Metax, Sanechips (ZhuFeng V1.0), Intel, Ascend, Kunpeng, and AMD.

## 🔥 Updates

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

<!-- * **Aug 28, 2024**: Support 1M context under the InternLM2.5-7B-Chat-1M model, utilizing 24GB of VRAM and 150GB of DRAM. The detailed tutorial is [here](./doc/en/long_context_tutorial.md). -->

## 🌟 Show Cases

### Local LLM Powerhouse: Run State-of-the-Art Models on Your Desktop

KTransformers empowers you to run powerful LLMs locally with impressive performance.  Here are some examples:

<p align="center">
  <picture>
    <img alt="VSCode Copilot Example" src="https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285" width=80%>
  </picture>
</p>

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4\_K\_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Performance Boost:**  Achieve significant speedups compared to llama.cpp:
        *   **Prefill:** Up to 27.79x faster (e.g., 255.26 tokens/s with optimized AMX-based MoE kernel).
        *   **Decode:** Up to 3.03x faster (e.g., 13.69 tokens/s with selective expert activation).
    *   **Open Source Roadmap:**  AMX optimizations and selective expert activation will be open-sourced in V0.3 (preview binary available).
*   **Local 236B DeepSeek-Coder-V2:** Run the Q4\_K\_M version on a local desktop using only 21GB VRAM and 136GB DRAM. This model outperforms GPT-4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).
    *   **Performance:** Prefill at 126 tokens/s (2K prompt) and generate at 13.6 tokens/s.
    *   **Integration:**  Compatible with OpenAI and Ollama APIs, enabling seamless integration with frontends like [Tabby](https://github.com/TabbyML/tabby).

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <img alt="VSCode Integration" src="https://github.com/user-attachments/assets/4c6a8a38-05aa-497d-8eb1-3a5b3918429c" width=80%>
  </picture>
</p>

<!--
### 1M Context with Limited Resources

KTransformers unlocks long context capabilities on your local machine:

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

## 🚀 Quick Start

Get up and running with KTransformers quickly!

### 📥 Installation

Follow the [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html) to get started.

## 📃 Brief Injection Tutorial

KTransformers employs a user-friendly, template-based injection framework, enabling researchers to swap out original Torch modules with optimized alternatives, as well as easily combine multiple optimizations to explore their synergistic effects.

<br>
<p align="center">
  <picture>
    <img alt="Injection Structure" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

KTransformers focuses on local deployments constrained by limited resources, especially heterogeneous computing opportunities, such as GPU/CPU offloading of quantized models. For example, we support efficient <a herf="https://github.com/Mozilla-Ocho/llamafile/tree/main">Llamafile</a> and <a herf="https://github.com/IST-DASLab/marlin">Marlin</a> kernels for CPU and GPU, respectively. More details can be found <a herf="doc/en/operators/llamafile.md">here</a>.

### Example Usage

To utilize provided kernels, create a YAML-based injection template and add `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

The `optimize_and_load_gguf` function iterates through the model's sub-modules, matches rules specified in your YAML rule file, and replaces them with advanced modules.
After injection, the original `generate` interface is available, alongside `prefill_and_generate`, which enables further optimizations like CUDAGraph.

### How to Customize Your Model

A detailed tutorial for injection and multi-GPU usage, using DeepSeek-V2 as an example, is available [here](doc/en/injection_tutorial.md).

Here is a YAML template example for replacing all original Linear modules with Marlin, an advanced 4-bit quantization kernel:

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

Each rule in the YAML file has two parts: `match` and `replace`. The `match` part specifies which module should be replaced, and the `replace` part specifies the module to be injected with initialization keywords.

Find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14 in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory, used in the `local_chat.py` demo.

Refer to the [design document](doc/en/deepseek-v2-injection.md) for details on the injection framework's design principles and implementation.

## <a id="ack"></a>Acknowledgment and Contributors

KTransformers builds upon the flexible Transformers framework and leverages advanced kernels such as GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer. We plan to contribute our modifications back to the community.

KTransformers is actively developed by contributors from the <a href="https://madsys.cs.tsinghua.edu.cn/">MADSys group</a> at Tsinghua University and members from <a href="http://approaching.ai/">Approaching.AI</a>. We welcome new contributors to help make KTransformers even better.

## <a id="discussion"></a>Discussion

Please open an issue for questions or join our WeChat group (QR Code below) for further discussion.

<p align="center">
<img src="WeChatGroup.png" width="20%">
</p>

## <a id="FAQ"></a>🙋 FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).