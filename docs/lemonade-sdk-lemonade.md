# Lemonade: Run Local LLMs with GPU and NPU Acceleration

[Lemonade](https://github.com/lemonade-sdk/lemonade) empowers you to run Large Language Models (LLMs) locally with optimized performance, leveraging your GPU and NPU hardware for blazing-fast inference.

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/5xXzkMu8Zk)
[![Tests](https://github.com/lemonade-sdk/lemonade/actions/workflows/test_lemonade.yml/badge.svg)](https://github.com/lemonade-sdk/lemonade/actions/workflows/test_lemonade.yml)
[![Windows 11](https://img.shields.io/badge/Windows-11-0078D6?logo=windows&logoColor=white)](docs/README.md#installation)
[![Ubuntu 24.04 & 25.04](https://img.shields.io/badge/Ubuntu-24.04%20%7C%2025.04-E95420?logo=ubuntu&logoColor=white)](https://lemonade-server.ai/#linux)
[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-blue?logo=python&logoColor=white)](docs/README.md#installation)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lemonade-sdk/lemonade/blob/main/docs/contribute.md)
[![Latest Release](https://img.shields.io/github/v/release/lemonade-sdk/lemonade?include_prereleases)](https://github.com/lemonade-sdk/lemonade/releases/latest)
[![GitHub Downloads](https://img.shields.io/github/downloads/lemonade-sdk/lemonade/total.svg)](https://tooomm.github.io/github-release-stats/?username=lemonade-sdk&repository=lemonade)
[![GitHub Issues](https://img.shields.io/github/issues/lemonade-sdk/lemonade)](https://github.com/lemonade-sdk/lemonade/issues)
[![License](https://img.shields.io/badge/License-Apache-yellow.svg)](https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Star History](https://img.shields.io/badge/Star%20History-View-brightgreen)](https://star-history.com/#lemonade-sdk/lemonade)

<p align="center">
  <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/banner.png?raw=true" alt="Lemonade Banner" />
</p>

<div align="center">
  <a href="https://lemonade-server.ai">Download</a> |
  <a href="https://lemonade-server.ai/docs/">Documentation</a> |
  <a href="https://discord.gg/5xXzkMu8Zk">Discord</a>
</div>

Lemonade provides a powerful and flexible platform for running LLMs locally, offering optimized performance through GPU and NPU acceleration.  Startups like Styrk AI, research teams like Hazy Research at Stanford, and companies like AMD use Lemonade to run LLMs.

## Key Features

*   **GPU and NPU Acceleration:**  Leverage your hardware for faster LLM inference.
*   **OpenAI Compatibility:** Seamlessly integrate with your favorite OpenAI-compatible applications.
*   **Model Management:** Easily download, manage, and switch between various LLM models.
*   **Cross-Platform Support:** Runs on Windows and Linux, with Python support.
*   **Built-in Chat Interface:** Start interacting with your LLMs immediately.
*   **Flexible Integration:** Use the Lemonade API and CLI for custom integrations.
*   **Open Source:**  Committed to open source and actively welcoming contributions.

## Getting Started: Quick Steps

<div align="center">

| Step 1: Download & Install | Step 2: Launch and Pull Models | Step 3: Start chatting! |
|:---------------------------:|:-------------------------------:|:------------------------:|
| <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/install.gif?raw=true" alt="Download & Install" width="245" /> | <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/launch_and_pull.gif?raw=true" alt="Launch and Pull Models" width="245" /> | <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/chat.gif?raw=true" alt="Start chatting!" width="245" /> |
|Install using a [GUI](https://github.com/lemonade-sdk/lemonade/releases/latest/download/Lemonade_Server_Installer.exe) (Windows only), [pip](https://lemonade-server.ai/install_options.html), or [from source](https://lemonade-server.ai/install_options.html). |Use the [Model Manager](#model-library) to install models|A built-in chat interface is available!|
</div>

### Integrate with Popular Apps

Use Lemonade with your preferred OpenAI-compatible apps!

<p align="center">
  <a href="https://lemonade-server.ai/docs/server/apps/open-webui/" title="Open WebUI" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/openwebui.jpg" alt="Open WebUI" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/continue/" title="Continue" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/continue_dev.png" alt="Continue" width="60" /></a>&nbsp;&nbsp;<a href="https://github.com/amd/gaia" title="Gaia" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/gaia.ico" alt="Gaia" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/anythingLLM/" title="AnythingLLM" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/anything_llm.png" alt="AnythingLLM" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/ai-dev-gallery/" title="AI Dev Gallery" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/ai_dev_gallery.webp" alt="AI Dev Gallery" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/lm-eval/" title="LM-Eval" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/lm_eval.png" alt="LM-Eval" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/codeGPT/" title="CodeGPT" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/codegpt.jpg" alt="CodeGPT" width="60" /></a>&nbsp;&nbsp;<a href="https://github.com/lemonade-sdk/lemonade/blob/main/docs/server/apps/ai-toolkit.md" title="AI Toolkit" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/ai_toolkit.png" alt="AI Toolkit" width="60" /></a>
</p>

> [!TIP]
> Want your app featured here? Let's do it! Shoot us a message on [Discord](https://discord.gg/5xXzkMu8Zk), [create an issue](https://github.com/lemonade-sdk/lemonade/issues), or [email](lemonade@amd.com).

## Using the CLI

To get started with the CLI:

```
lemonade-server run Gemma-3-4b-it-GGUF
```

To install models before running:

```
lemonade-server pull Gemma-3-4b-it-GGUF
```

See all available models:

```
lemonade-server list
```

> **Note**:  If you installed from source, use the `lemonade-server-dev` command instead.

> **Tip**: You can use `--llamacpp vulkan/rocm` to select a backend when running GGUF models.

## Model Library

Lemonade supports GGUF and ONNX models.  Detailed information is available in the [Supported Configuration](#supported-configurations) section. Find all built-in models [here](https://lemonade-server.ai/docs/server/server_models/).

Use the [Model Manager](http://localhost:8000/#model-management) (requires server to be running) to import custom GGUF and ONNX models from Hugging Face.

<p align="center">
  <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/model_manager.png?raw=true" alt="Model Manager" width="650" />
</p>

## Supported Configurations

Lemonade offers various configuration options, easily changeable at runtime.  Learn more [here](./docs/README.md#software-and-hardware-overview).

| Hardware | Engine: OGA | Engine: llamacpp | Engine: HF | Windows | Linux |
|----------|-------------|------------------|------------|---------|-------|
| **🧠 CPU** | All platforms | All platforms | All platforms | ✅ | ✅ |
| **🎮 GPU** | — | Vulkan: All platforms<br>ROCm: Selected AMD platforms* | — | ✅ | ✅ |
| **🤖 NPU** | AMD Ryzen™ AI 300 series | — | — | ✅ | — |

<details>
<summary><small><i>* See supported AMD ROCm platforms</i></small></summary>

<br>

<table>
  <thead>
    <tr>
      <th>Architecture</th>
      <th>Platform Support</th>
      <th>GPU Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>gfx1151</b> (STX Halo)</td>
      <td>Windows, Ubuntu</td>
      <td>Ryzen AI MAX+ Pro 395</td>
    </tr>
    <tr>
      <td><b>gfx120X</b> (RDNA4)</td>
      <td>Windows, Ubuntu</td>
      <td>Radeon AI PRO R9700, RX 9070 XT/GRE/9070, RX 9060 XT</td>
    </tr>
    <tr>
      <td><b>gfx110X</b> (RDNA3)</td>
      <td>Windows, Ubuntu</td>
      <td>Radeon PRO W7900/W7800/W7700/V710, RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT</td>
    </tr>
  </tbody>
</table>
</details>

## Integrate with Your Application

Use OpenAI-compatible client libraries by setting the base URL to `http://localhost:8000/api/v1`.

| Python | C++ | Java | C# | Node.js | Go | Ruby | Rust | PHP |
|--------|-----|------|----|---------|----|-------|------|-----|
| [openai-python](https://github.com/openai/openai-python) | [openai-cpp](https://github.com/olrea/openai-cpp) | [openai-java](https://github.com/openai/openai-java) | [openai-dotnet](https://github.com/openai/openai-dotnet) | [openai-node](https://github.com/openai/openai-node) | [go-openai](https://github.com/sashabaranov/go-openai) | [ruby-openai](https://github.com/alexrudall/ruby-openai) | [async-openai](https://github.com/64bit/async-openai) | [openai-php](https://github.com/openai-php/client) |

### Python Client Example
```python
from openai import OpenAI

# Initialize the client to use Lemonade Server
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="lemonade"  # required but unused
)

# Create a chat completion
completion = client.chat.completions.create(
    model="Llama-3.2-1B-Instruct-Hybrid",  # or any other available model
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# Print the response
print(completion.choices[0].message.content)
```

For detailed instructions, see the [Integration Guide](./docs/server/server_integration.md).

## Beyond an LLM Server

The [Lemonade SDK](./docs/README.md) provides:

*   🐍 **[Lemonade API](./docs/lemonade_api.md)**: A Python API for seamless LLM integration.
*   🖥️ **[Lemonade CLI](./docs/dev_cli/README.md)**: Use the CLI to test LLMs, benchmark performance, and profile memory usage.

## Frequently Asked Questions

Find answers to your questions in our [FAQ Guide](./docs/faq.md).

## Contribute

We welcome contributions! Review our [contribution guide](./docs/contribute.md) and find beginner-friendly issues tagged with "Good First Issue."

<a href="https://github.com/lemonade-sdk/lemonade/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
  <img src="https://img.shields.io/badge/🍋Lemonade-Good%20First%20Issue-yellowgreen?colorA=38b000&colorB=cccccc" alt="Good First Issue" />
</a>

## Maintainers

This project is sponsored by AMD and maintained by @danielholanda, @jeremyfowers, @ramkrishna, and @vgodsoe. Reach out via [issue](https://github.com/lemonade-sdk/lemonade/issues), email (lemonade@amd.com), or Discord.

## License and Attribution

*   Built with Python for the open source community.
*   Uses tools from ggml/llama.cpp, OnnxRuntime GenAI, Hugging Face Hub, OpenAI API, and more.
*   Accelerated by mentorship from the OCV Catalyst program.
*   Licensed under the [Apache 2.0 License](https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE).
    *   Portions of the project are licensed as described in [NOTICE.md](./NOTICE.md).