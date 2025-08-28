# 🍋 Lemonade: Run LLMs Locally with GPU and NPU Acceleration

**Supercharge your local LLM performance with Lemonade, the open-source server engineered for speed and efficiency.** ([See the original repo](https://github.com/lemonade-sdk/lemonade))

[![Discord](https://img.shields.io/discord/1175085518182716456?label=Discord&logo=discord&logoColor=white)](https://discord.gg/5xXzkMu8Zk)
[![Tests](https://github.com/lemonade-sdk/lemonade/actions/workflows/test_lemonade.yml/badge.svg)](https://github.com/lemonade-sdk/lemonade/actions/workflows/test_lemonade.yml)
[![Windows 11](https://img.shields.io/badge/Windows-11-0078D6?logo=windows&logoColor=white)](docs/README.md#installation)
[![Ubuntu 24.04 | 25.04](https://img.shields.io/badge/Ubuntu-24.04%20%7C%2025.04-E95420?logo=ubuntu&logoColor=white)](https://lemonade-server.ai/#linux)
[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-blue?logo=python&logoColor=white)](docs/README.md#installation)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lemonade-sdk/lemonade/blob/main/docs/contribute.md)
[![Latest Release](https://img.shields.io/github/v/release/lemonade-sdk/lemonade?include_prereleases)](https://github.com/lemonade-sdk/lemonade/releases/latest)
[![GitHub Downloads](https://img.shields.io/github/downloads/lemonade-sdk/lemonade/total.svg)](https://tooomm.github.io/github-release-stats/?username=lemonade-sdk&repository=lemonade)
[![GitHub issues](https://img.shields.io/github/issues/lemonade-sdk/lemonade)](https://github.com/lemonade-sdk/lemonade/issues)
[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Star History Chart](https://img.shields.io/badge/Star%20History-View-brightgreen)](https://star-history.com/#lemonade-sdk/lemonade)

<p align="center">
  <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/banner.png?raw=true" alt="Lemonade Banner" />
</p>

<div align="center">
  <a href="https://lemonade-server.ai">Download</a> |
  <a href="https://lemonade-server.ai/docs/">Documentation</a> |
  <a href="https://discord.gg/5xXzkMu8Zk">Discord</a>
</div>

Lemonade empowers you to run large language models (LLMs) locally with exceptional performance, leveraging the power of your GPU and NPU. Built with state-of-the-art inference engines, Lemonade is the preferred choice for startups, research teams, and large enterprises looking to optimize their LLM deployments.

**Key Features:**

*   **GPU and NPU Acceleration:** Maximize performance on your hardware.
*   **Easy Integration:** Compatible with OpenAI-compatible client libraries.
*   **Model Manager:** Simplify model downloads and management.
*   **Multiple Engine Support:** Seamlessly switch between different inference engines (OGA, llamacpp, HF) at runtime.
*   **Built-in Chat Interface:** Start chatting with your models immediately.
*   **OpenAI API Compatibility:** Works with popular OpenAI-compatible applications.

**Used by:**

*   [Styrk AI](https://styrk.ai/styrk-ai-and-amd-guardrails-for-your-on-device-ai-revolution/)
*   [Hazy Research at Stanford](https://www.amd.com/en/developer/resources/technical-articles/2025/minions--on-device-and-cloud-language-model-collaboration-on-ryz.html)
*   [AMD](https://www.amd.com/en/developer/resources/technical-articles/unlocking-a-wave-of-llm-apps-on-ryzen-ai-through-lemonade-server.html)

## Getting Started

Follow these simple steps to get up and running:

<div align="center">

| Step 1: Download & Install | Step 2: Launch and Pull Models | Step 3: Start chatting! |
|:---------------------------:|:-------------------------------:|:------------------------:|
| <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/install.gif?raw=true" alt="Download & Install" width="245" /> | <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/launch_and_pull.gif?raw=true" alt="Launch and Pull Models" width="245" /> | <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/chat.gif?raw=true" alt="Start chatting!" width="245" /> |
|Install using a [GUI](https://github.com/lemonade-sdk/lemonade/releases/latest/download/Lemonade_Server_Installer.exe) (Windows only), [pip](https://lemonade-server.ai/install_options.html), or [from source](https://lemonade-server.ai/install_options.html). |Use the [Model Manager](#model-library) to install models|A built-in chat interface is available!|
</div>

### Seamlessly Integrate with Your Favorite Tools

Lemonade is compatible with numerous OpenAI-compatible applications.  Get started today with your favorite LLM tools:

<p align="center">
  <a href="https://lemonade-server.ai/docs/server/apps/open-webui/" title="Open WebUI" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/openwebui.jpg" alt="Open WebUI" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/continue/" title="Continue" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/continue_dev.png" alt="Continue" width="60" /></a>&nbsp;&nbsp;<a href="https://github.com/amd/gaia" title="Gaia" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/gaia.ico" alt="Gaia" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/anythingLLM/" title="AnythingLLM" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/anything_llm.png" alt="AnythingLLM" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/ai-dev-gallery/" title="AI Dev Gallery" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/ai_dev_gallery.webp" alt="AI Dev Gallery" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/lm-eval/" title="LM-Eval" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/lm_eval.png" alt="LM-Eval" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/codeGPT/" title="CodeGPT" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/codegpt.jpg" alt="CodeGPT" width="60" /></a>&nbsp;&nbsp;<a href="https://github.com/lemonade-sdk/lemonade/blob/main/docs/server/apps/ai-toolkit.md" title="AI Toolkit" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/ai_toolkit.png" alt="AI Toolkit" width="60" /></a>
</p>

> [!TIP]
> Want your app featured here? Let's do it! Shoot us a message on [Discord](https://discord.gg/5xXzkMu8Zk), [create an issue](https://github.com/lemonade-sdk/lemonade/issues), or [email](lemonade@amd.com).

## Command Line Interface (CLI)

Quickly run and manage your LLMs using the Lemonade CLI.

To run and chat with Gemma 3:

```bash
lemonade-server run Gemma-3-4b-it-GGUF
```

To install models ahead of time, use the `pull` command:

```bash
lemonade-server pull Gemma-3-4b-it-GGUF
```

To check all models available, use the `list` command:

```bash
lemonade-server list
```

> **Note**:  If you installed from source, use the `lemonade-server-dev` command instead.

> **Tip**: You can use `--llamacpp vulkan/rocm` to select a backend when running GGUF models.

## Model Library

Lemonade supports both GGUF and ONNX models, offering flexibility in your model selection. Find out more details in the [Supported Configuration](#supported-configurations) section. Browse the list of built-in models [here](https://lemonade-server.ai/docs/server/server_models/).

Easily import custom GGUF and ONNX models from Hugging Face using our [Model Manager](http://localhost:8000/#model-management) (requires server to be running).

<p align="center">
  <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/model_manager.png?raw=true" alt="Model Manager" width="650" />
</p>

## Supported Configurations

Lemonade is designed to work with a wide range of configurations, enabling you to switch between them at runtime easily. Learn more about it [here](./docs/README.md#software-and-hardware-overview).

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

Seamlessly integrate Lemonade Server into your application using any OpenAI-compatible client library, configured to use `http://localhost:8000/api/v1` as the base URL.  Here's how in Python:

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

Find detailed integration instructions in the [Integration Guide](./docs/server/server_integration.md).

## Beyond the Server: The Lemonade SDK

The Lemonade SDK goes beyond a simple LLM server, offering these key components:

*   🐍 **[Lemonade API](./docs/lemonade_api.md)**: A high-level Python API for direct integration of Lemonade LLMs into Python applications.
*   🖥️ **[Lemonade CLI](./docs/dev_cli/README.md)**:  The `lemonade` CLI allows you to test, benchmark, and profile LLMs (ONNX, GGUF, SafeTensors) to characterize your models on your hardware.

## Frequently Asked Questions (FAQ)

Find answers to common questions in our [FAQ Guide](./docs/faq.md).

## Contributing

We welcome contributions!  Please review our [contribution guide](./docs/contribute.md) to get started.

Find beginner-friendly issues tagged "Good First Issue" to start contributing:

<a href="https://github.com/lemonade-sdk/lemonade/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
  <img src="https://img.shields.io/badge/🍋Lemonade-Good%20First%20Issue-yellowgreen?colorA=38b000&colorB=cccccc" alt="Good First Issue" />
</a>

## Project Maintainers

Lemonade is proudly sponsored by AMD and maintained by @danielholanda @jeremyfowers @ramkrishna @vgodsoe.  Contact us by:

*   Filing an [issue](https://github.com/lemonade-sdk/lemonade/issues)
*   Emailing [lemonade@amd.com](mailto:lemonade@amd.com)
*   Joining our [Discord](https://discord.gg/5xXzkMu8Zk)

## License and Attribution

This project is:

*   [Built with Python](https://www.amd.com/en/developer/resources/technical-articles/2025/rethinking-local-ai-lemonade-servers-python-advantage.html) for the open-source community.
*   Powered by great tools from:
    *   [ggml/llama.cpp](https://github.com/ggml-org/llama.cpp)
    *   [OnnxRuntime GenAI](https://github.com/microsoft/onnxruntime-genai)
    *   [Hugging Face Hub](https://github.com/huggingface/huggingface_hub)
    *   [OpenAI API](https://github.com/openai/openai-python)
    *   and more...
*   Accelerated by the OCV Catalyst program.
*   Licensed under the [Apache 2.0 License](https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE).
    *   Parts of the project are licensed as described in [NOTICE.md](./NOTICE.md).

<!--This file was originally licensed under Apache 2.0. It has been modified.
Modifications Copyright (c) 2025 AMD-->