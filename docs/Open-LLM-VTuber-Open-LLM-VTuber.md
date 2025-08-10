<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion, Always by Your Side</h1>

<h3 align="center">
    <a href="https://github.com/t41372/Open-LLM-VTuber" target="_blank">
        <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="GitHub Stars">
    </a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/releases">
        <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber" alt="GitHub Release">
    </a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber" alt="License">
    </a>
    <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml">
        <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">
    </a>
    <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml">
        <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff">
    </a>
    <a href="https://hub.docker.com/r/t41372/open-llm-vtuber">
        <img src="https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue" alt="Docker">
    </a>
    <a href="https://qm.qq.com/q/ngvNUQpuKI">
        <img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white" alt="QQ Group">
    </a>
    <a href="https://pd.qq.com/s/tt54r3bu">
        <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white" alt="QQ Channel (Dev)">
    </a>
    <a href="https://www.buymeacoffee.com/yi.ting">
        <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
    </a>
    <a href="https://discord.gg/3UDA8YFDXx">
        <img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord Server">
    </a>
    <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber">
        <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
    </a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md">
        <img src="https://img.shields.io/badge/README-中文-blue" alt="Chinese README">
    </a>
    <a href="https://open-llm-vtuber.github.io/docs/quick-start">
        <img src="https://img.shields.io/badge/Documentation-Available-green" alt="Documentation">
    </a>
    <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2">
        <img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow" alt="Roadmap">
    </a>
    <a href="https://trendshift.io/repositories/12358" target="_blank">
        <img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
    </a>
</h3>

**Open-LLM-VTuber is your open-source, voice-interactive AI companion, bringing a Live2D avatar to life on your desktop or in your browser!**

*   [View on GitHub](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

> **Note:** This project is actively under development.

## 🔑 Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Offline Mode:** Utilize local models for complete privacy and independence from the internet.
*   **Web and Desktop Clients:** Choose your preferred interaction method, with a unique "desktop pet mode".
*   **Advanced Interactions:**
    *   Visual perception (camera, screen recording, screenshots).
    *   Voice interruption without headphones.
    *   Touch feedback.
    *   Live2D expression mapping.
    *   "Pet mode" with transparent background for desktop presence.
    *   Display AI's inner thoughts.
    *   AI proactive speaking feature
    *   Persistent chat logs.
    *   TTS translation support.
*   **Extensive Model Support:** Broad compatibility with various LLMs (Ollama, OpenAI, Gemini, etc.), ASR (sherpa-onnx, Faster-Whisper, etc.), and TTS (sherpa-onnx, Coqui-TTS, etc.).
*   **Highly Customizable:**
    *   Simple module configuration.
    *   Character customization (Live2D models, prompts, voice cloning).
    *   Flexible Agent implementation.
    *   Extensible design for easy integration of new LLMs, ASR, and TTS.

## 🖼️ Demo

| ![](./assets/i1.jpg) | ![](./assets/i2.jpg) |
|:---:|:---:|
| ![](./assets/i3.jpg) | ![](./assets/i4.jpg) |

## 🚀 Quick Start

Get started with your AI companion! Follow the [Quick Start Guide](https://open-llm-vtuber.github.io/docs/quick-start) in our documentation.

## ☝ Update

Use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## 😢 Uninstall

Most files, including Python dependencies and models, are stored in the project folder.

However, models downloaded via ModelScope or Hugging Face may also be in `MODELSCOPE_CACHE` or `HF_HOME`. While we aim to keep them in the project's `models` directory, it's good to double-check.  

Review the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`.  

## 🤗 Contribute

Learn how you can contribute to the project by checking out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## 🎉 Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant): Your Live2D desktop assistant powered by LLM!

## 📜 Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

Thank you to all our contributors!

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)