<!--
SPDX-License-Identifier: Apache-2.0
-->

# Lemonade: Run LLMs Locally with GPU and NPU Acceleration

**Supercharge your local Large Language Model (LLM) performance with Lemonade, leveraging the power of your GPU and NPU!**  [View on GitHub](https://github.com/lemonade-sdk/lemonade)

<p align="center">
  <a href="https://discord.gg/5xXzkMu8Zk">
    <img src="https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white" alt="Discord" />
  </a>
  <a href="https://github.com/lemonade-sdk/lemonade/tree/main/test" title="Check out our tests">
    <img src="https://github.com/github/workflows/blob/main/test_lemonade.yml/badge.svg" alt="Lemonade tests" />
  </a>
  <a href="docs/README.md#installation" title="Check out our instructions">
    <img src="https://img.shields.io/badge/Windows-11-0078D6?logo=windows&logoColor=white" alt="Windows 11" />
  </a>
  <a href="https://lemonade-server.ai/#linux" title="Ubuntu 24.04 & 25.04 Supported">
    <img src="https://img.shields.io/badge/Ubuntu-24.04%20%7C%2025.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu 24.04 | 25.04" />
  </a>
  <a href="docs/README.md#installation" title="Check out our instructions">
    <img src="https://img.shields.io/badge/Python-3.10--3.13-blue?logo=python&logoColor=white" alt="Made with Python" />
  </a>
  <a href="https://github.com/lemonade-sdk/lemonade/blob/main/docs/contribute.md" title="Contribution Guide">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" />
  </a>
  <a href="https://github.com/lemonade-sdk/lemonade/releases/latest" title="Download the latest release">
    <img src="https://img.shields.io/github/v/release/lemonade-sdk/lemonade?include_prereleases" alt="Latest Release" />
  </a>
  <a href="https://tooomm.github.io/github-release-stats/?username=lemonade-sdk&repository=lemonade">
    <img src="https://img.shields.io/github/downloads/lemonade-sdk/lemonade/total.svg" alt="GitHub downloads" />
  </a>
  <a href="https://github.com/lemonade-sdk/lemonade/issues">
    <img src="https://img.shields.io/github/issues/lemonade-sdk/lemonade" alt="GitHub issues" />
  </a>
  <a href="https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache-yellow.svg" alt="License: Apache" />
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
  </a>
  <a href="https://star-history.com/#lemonade-sdk/lemonade">
    <img src="https://img.shields.io/badge/Star%20History-View-brightgreen" alt="Star History Chart" />
  </a>
</p>
<p align="center">
  <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/banner.png?raw=true" alt="Lemonade Banner" />
</p>
<h3 align="center">
  <a href="https://lemonade-server.ai">Download</a> |
  <a href="https://lemonade-server.ai/docs/">Documentation</a> |
  <a href="https://discord.gg/5xXzkMu8Zk">Discord</a>
</h3>

Lemonade empowers developers and researchers to run LLMs locally with optimal performance by configuring state-of-the-art inference engines for their NPUs and GPUs. Trusted by startups, research teams, and industry leaders like AMD.

**Key Features:**

*   **GPU & NPU Acceleration:** Harness the power of your hardware for faster LLM inference.
*   **OpenAI-Compatible API:** Seamlessly integrate Lemonade with your favorite OpenAI-compatible applications.
*   **Model Manager:** Easily download and manage GGUF and ONNX models.
*   **Cross-Platform Support:** Works on Windows and Linux.
*   **Flexible Backends:** Switch between CPU, GPU (Vulkan/ROCm), and NPU (AMD Ryzen AI) at runtime.
*   **Python API and CLI:**  Integrate directly into Python applications or use the command line interface.
*   **Extensive Model Support:** Support for GGUF and ONNX models with built-in models available.

## Getting Started: Run Your First LLM

<div align="center">

| Step 1: Download & Install | Step 2: Launch and Pull Models | Step 3: Start chatting! |
|:---------------------------:|:-------------------------------:|:------------------------:|
| <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/install.gif?raw=true" alt="Download & Install" width="245" /> | <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/launch_and_pull.gif?raw=true" alt="Launch and Pull Models" width="245" /> | <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/chat.gif?raw=true" alt="Start chatting!" width="245" /> |
|Install using a [GUI](https://github.com/lemonade-sdk/lemonade/releases/latest/download/Lemonade_Server_Installer.exe) (Windows only), [pip](https://lemonade-server.ai/install_options.html), or [from source](https://lemonade-server.ai/install_options.html). |Use the [Model Manager](#model-library) to install models|A built-in chat interface is available!|
</div>

### Integrate with Your Favorite Tools

Lemonade works seamlessly with popular OpenAI-compatible applications:

<p align="center">
  <a href="https://lemonade-server.ai/docs/server/apps/open-webui/" title="Open WebUI" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/openwebui.jpg" alt="Open WebUI" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/continue/" title="Continue" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/continue_dev.png" alt="Continue" width="60" /></a>&nbsp;&nbsp;<a href="https://github.com/amd/gaia" title="Gaia" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/gaia.ico" alt="Gaia" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/anythingLLM/" title="AnythingLLM" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/anything_llm.png" alt="AnythingLLM" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/ai-dev-gallery/" title="AI Dev Gallery" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/ai_dev_gallery.webp" alt="AI Dev Gallery" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/lm-eval/" title="LM-Eval" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/lm_eval.png" alt="LM-Eval" width="60" /></a>&nbsp;&nbsp;<a href="https://lemonade-server.ai/docs/server/apps/codeGPT/" title="CodeGPT" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/codegpt.jpg" alt="CodeGPT" width="60" /></a>&nbsp;&nbsp;<a href="https://github.com/lemonade-sdk/lemonade/blob/main/docs/server/apps/ai-toolkit.md" title="AI Toolkit" target="_blank"><img src="https://raw.githubusercontent.com/lemonade-sdk/assets/refs/heads/main/partner_logos/ai_toolkit.png" alt="AI Toolkit" width="60" /></a>
</p>

> [!TIP]
> Want your app featured here? Let's do it! Shoot us a message on [Discord](https://discord.gg/5xXzkMu8Zk), [create an issue](https://github.com/lemonade-sdk/lemonade/issues), or [email](lemonade@amd.com).

## Command-Line Interface (CLI)

Run LLMs with ease using the `lemonade-server` CLI:

```bash
lemonade-server run Gemma-3-4b-it-GGUF
```

Install models:

```bash
lemonade-server pull Gemma-3-4b-it-GGUF
```

List available models:

```bash
lemonade-server list
```

> **Note**: Use `lemonade-server-dev` if you installed from source.
> **Tip**: Use `--llamacpp vulkan/rocm` to select a backend when running GGUF models.

## Model Library

Lemonade supports GGUF and ONNX models. Explore built-in models [here](https://lemonade-server.ai/docs/server/server_models/).

Import custom models from Hugging Face using the [Model Manager](http://localhost:8000/#model-management) (server must be running).

<p align="center">
  <img src="https://github.com/lemonade-sdk/assets/blob/main/docs/model_manager.png?raw=true" alt="Model Manager" width="650" />
</p>

## Supported Configurations

Lemonade offers versatile configurations for optimal performance:

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
      <td>Windows only</td>
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

## Integrate Lemonade Server into Your Application

Use any OpenAI-compatible client library:

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

## Beyond the Server

The [Lemonade SDK](./docs/README.md) also includes:

*   🐍 **[Lemonade API](./docs/lemonade_api.md)**: Python API for LLM integration.
*   🖥️ **[Lemonade CLI](./docs/dev_cli/README.md)**: Command-line tools for model testing and benchmarking.

## FAQ

Find answers to frequently asked questions in our [FAQ Guide](./docs/faq.md).

## Contributing

We welcome contributions! See our [contribution guide](./docs/contribute.md) to get started.

Find beginner-friendly issues tagged "Good First Issue":

<a href="https://github.com/lemonade-sdk/lemonade/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
  <img src="https://img.shields.io/badge/🍋Lemonade-Good%20First%20Issue-yellowgreen?colorA=38b000&colorB=cccccc" alt="Good First Issue" />
</a>

## Maintainers

This project is sponsored by AMD and maintained by @danielholanda @jeremyfowers @ramkrishna @vgodsoe. Contact us via [issues](https://github.com/lemonade-sdk/lemonade/issues), [email](mailto:lemonade@amd.com), or [Discord](https://discord.gg/5xXzkMu8Zk).

## License and Attribution

*   Built with Python ❤️ for the open source community.
*   Leveraging tools from: ggml/llama.cpp, OnnxRuntime GenAI, Hugging Face Hub, OpenAI API, and more...
*   Accelerated by mentorship from the OCV Catalyst program.
*   Licensed under the [Apache 2.0 License](https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE).
    *   Portions licensed as described in [NOTICE.md](./NOTICE.md).

<!--This file was originally licensed under Apache 2.0. It has been modified.
Modifications Copyright (c) 2025 AMD-->
```
Key improvements and SEO considerations:

*   **Hook:**  A compelling one-sentence hook to grab the reader's attention immediately.
*   **Target Keywords:**  Includes terms like "local LLMs," "GPU acceleration," "NPU acceleration," "LLM inference," and "OpenAI-compatible" to improve search visibility.
*   **Clear Headings:** Uses `h2` and `h3` tags for a structured and readable document, aiding SEO and user experience.
*   **Bulleted Key Features:** Highlights the main benefits in an easily digestible format.
*   **Concise Language:**  Rephrases content for clarity and conciseness.
*   **Call to Actions:** Encourages users to download, explore the documentation, and join the community.
*   **Internal Links:** Includes links to different sections of the documentation.
*   **External Links:** Properly links to related projects, the Discord server, and the official website.
*   **Alt Text:** Adds alt text to images for accessibility and SEO.
*   **Maintainer Information:** Provides clear contact information for support and contributions.
*   **License Information:** Clearly states the license.
*   **OpenAI Compatibility:** The section on OpenAI compatibility is highlighted to emphasize ease of integration.