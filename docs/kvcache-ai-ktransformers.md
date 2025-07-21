<div align="center">
  <picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
  </picture>
  <h3>Supercharge your LLM inference with KTransformers, a flexible framework for cutting-edge optimizations!</h3>
  <strong><a href="#features">✨ Key Features</a> | <a href="#show-cases">🌟 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#tutorial">📃 Tutorial</a> | <a href="https://github.com/kvcache-ai/ktransformers/discussions">💬  Discussion </a> | <a href="#FAQ">🙋 FAQ</a></strong>
</div>

## Key Features

*   **Optimized Performance:** Experience significant speedups in LLM inference through advanced kernel optimizations, placement, and parallelism strategies.
*   **Flexible Architecture:** Built with extensibility at its core, KTransformers allows easy integration of optimized modules with a single line of code.
*   **Transformers Compatibility:** Seamlessly integrates with the Hugging Face Transformers library, providing a familiar interface.
*   **API Support:** Offers RESTful APIs compatible with OpenAI and Ollama, along with a user-friendly web UI.
*   **Heterogeneous Computing:** Supports GPU/CPU offloading for quantized models.
*   **Long Context Support:** Handle extended context lengths, enabling processing of larger inputs.
*   **Wide Hardware Support:** Works across various hardware vendors (Metax, Sanechips, Intel, Ascend, Kunpeng, AMD)

## Show Cases

KTransformers unlocks impressive performance gains, even on resource-constrained hardware.

*   **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Run the Q4_K_M version using only 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
    *   **Up to 27.79x speedup** compared to llama.cpp for prefill, and **3.03x speedup** for decode.
*   **Local 236B DeepSeek-Coder-V2:**  Run the Q4_K_M version using only 21GB VRAM and 136GB DRAM, scoring even better than GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).
    *   Faster Speed: 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation.
    *   VSCode Integration: Integrated with OpenAI and Ollama API for frontends such as Tabby.

<p align="center">
  <picture>
    <img alt="DeepSeek-Coder-V2 Score" src="https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693" width=100%>
  </picture>
</p>

_More advanced features will be coming soon. Stay tuned!_

## Quick Start

Get up and running with KTransformers in minutes!

### 📥 Installation

For detailed installation instructions, please refer to the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

## Tutorial: Implementing LLM Inference Optimizations

KTransformers provides a user-friendly injection framework, enabling researchers to easily replace original Torch modules with optimized variants and explore their synergistic effects.  This framework focuses on local deployments and leverages heterogeneous computing opportunities, such as GPU/CPU offloading of quantized models.

<p align="center">
  <picture>
    <img alt="Inject-Struction" src="https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e" width=65%>
  </picture>
</p>

### Example Usage

Create a YAML-based injection template and call `optimize_and_load_gguf` before using the Transformers model:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

### Customizing Your Model

Use YAML templates to specify module replacements.  For example, to replace all Linear modules with Marlin:

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

Detailed tutorials for DeepSeek-V2 and Qwen2-57B-A14, including the injection and multi-GPU setup, can be found in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory.

## Acknowledgments and Contributors

KTransformers builds on the foundation of the Hugging Face Transformers library. It also benefits from advanced kernels like GGUF/GGML, Llamafile, Marlin, sglang, and flashinfer.

KTransformers is actively maintained by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University and members from [Approaching.AI](http://approaching.ai/).

## Discussion

Have questions?  Open an issue or join our WeChat group using the QR code below.

<img src="WeChatGroup.png" alt="WeChat Group QR Code" width="200"/>

## FAQ

Find answers to common questions in the [FAQ](doc/en/FAQ.md).

[Back to top](#)
```
Key improvements and summary of changes:

*   **SEO Optimization:**
    *   Added a compelling one-sentence hook at the beginning.
    *   Used relevant keywords (LLM, inference, optimization, Transformers).
    *   Optimized headings for readability and search engines.
*   **Improved Readability:**
    *   Organized content with clear headings and subheadings.
    *   Used bullet points for key features.
    *   Simplified language.
*   **Summarization:**
    *   Condensed the original text while preserving key information.
    *   Removed redundant phrases and details.
*   **Structure:**
    *   Maintained the original section structure but streamlined the content.
*   **Added Value:**
    *   Included an "Acknowledgment" section.
    *   Re-integrated the QR code from the original.
    *   Included the "Back to top" link.
*   **Links:** Maintained the link back to the original repo.
*   **Cleaned up attachments** Removed unnecessary attachments to shorten and clean up the markdown.

This improved README is more concise, user-friendly, and better optimized for search engines, making it easier for potential users to understand and adopt KTransformers.