<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion, Brought to Life!</h1>

<h3 align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber" target="_blank">
    <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber?style=flat-square" alt="GitHub release">
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber?style=flat-square" alt="License">
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml" target="_blank">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml" target="_blank">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff">
  </a>
  <a href="https://hub.docker.com/r/t41372/open-llm-vtuber" target="_blank">
    <img src="https://img.shields.io/badge/Docker-t41372%2FOpen--LLM--VTuber-0db7ed?logo=docker&logoColor=blue&labelColor=white&color=blue&style=flat-square" alt="Docker">
  </a>
  <a href="https://qm.qq.com/q/ngvNUQpuKI" target="_blank">
    <img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Group">
  </a>
  <a href="https://pd.qq.com/s/tt54r3bu" target="_blank">
    <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Channel (dev)">
  </a>
  <br>
  <a href="https://www.buymeacoffee.com/yi.ting" target="_blank">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat-square&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
  </a>
  <a href="https://discord.gg/3UDA8YFDXx" target="_blank">
    <img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord Server">
  </a>
  <br>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
  <a href="https://open-llm-vtuber.github.io/docs/quick-start" target="_blank">
    Documentation
  </a>
  <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2" target="_blank">
    <img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow?style=flat-square" alt="Roadmap">
  </a>
  <br>
  <a href="https://trendshift.io/repositories/12358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55">
  </a>
</h3>

> **Open-LLM-VTuber** is an open-source project transforming your AI companion dreams into reality with a voice-interactive, customizable Live2D avatar.

<br>
<hr>

## Key Features

*   🖥️ **Cross-Platform Compatibility:** Seamlessly runs on Windows, macOS, and Linux, with support for both NVIDIA and non-NVIDIA GPUs, and CPU fallback.
*   🔒 **Offline Mode:** Interact with your AI companion entirely offline, ensuring data privacy and security.
*   💻 **Web & Desktop Clients:** Enjoy the flexibility of both web and desktop client modes, featuring an interactive experience with a transparent background desktop pet mode.
*   🎯 **Advanced Interaction:**
    *   👁️ Visual perception via camera, screen recording and screenshots.
    *   🎤 Noise-cancellation (AI doesn't hear itself).
    *   🫱 Touch feedback.
    *   😊 Live2D expressions.
    *   🐱 Desktop Pet Mode with transparent background and click-through.
    *   💭 AI's inner thoughts display.
    *   🗣️ Proactive AI speaking.
    *   💾 Chat log persistence.
    *   🌍 TTS translation support (e.g., chat in Chinese while AI uses Japanese voice).
*   🧠 **Extensive Model Support:** Broad compatibility with LLMs (Ollama, OpenAI, Gemini, Claude, etc.), ASRs (sherpa-onnx, Whisper.cpp, etc.), and TTS engines (sherpa-onnx, Coqui-TTS, Edge TTS, etc.).
*   🔧 **Highly Customizable:**
    *   ⚙️ Configuration-based module switching.
    *   🎨 Character customization with custom Live2D models, prompts, and voice cloning.
    *   🧩 Flexible Agent implementation to integrate any Agent architecture.
    *   🔌 Modular design for easy extension with custom LLMs, ASRs, and TTS modules.

## 🚀 Quick Start

Get started with your AI companion by following the installation guide in our [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) documentation.

## 🖼️ Demo

| ![](./assets/i1.jpg) | ![](./assets/i2.jpg) |
|:---:|:---:|
| ![](./assets/i3.jpg) | ![](./assets/i4.jpg) |

<hr>

## ☝ Update

Please use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## 😢 Uninstall

Most files, including Python dependencies and models, are stored in the project folder.

However, models downloaded via ModelScope or Hugging Face may also be in `MODELSCOPE_CACHE` or `HF_HOME`. While we aim to keep them in the project's `models` directory, it's good to double-check.  

Review the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`.  

## 🤗 Want to contribute?

Explore our [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to learn how you can contribute to the project.

<hr>
# 🎉🎉🎉 Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)
- Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

<hr>

## 📜 Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.


## Contributors

We are grateful to our contributors and maintainers for making this project possible.

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)

<hr>

**[Go back to the original repo](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)**