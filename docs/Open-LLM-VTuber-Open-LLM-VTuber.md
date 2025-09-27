<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion for Engaging Conversations</h1>

<p align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber" target="_blank">
    <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="Stars">
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber" alt="GitHub release">
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber" alt="License">
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml" target="_blank">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml" target="_blank">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff">
  </a>
  <a href="https://hub.docker.com/r/t41372/open-llm-vtuber" target="_blank">
    <img src="https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue" alt="Docker">
  </a>
  <a href="https://qm.qq.com/q/ngvNUQpuKI" target="_blank">
    <img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white" alt="QQ Group">
  </a>
  <a href="https://pd.qq.com/s/tt54r3bu" target="_blank">
    <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white" alt="QQ Channel">
  </a>
  <a href="https://www.buymeacoffee.com/yi.ting" target="_blank">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee">
  </a>
  <a href="https://discord.gg/3UDA8YFDXx" target="_blank">
    <img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord Server">
  </a>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
  <a href="https://open-llm-vtuber.github.io/docs/quick-start" target="_blank">
    <img src="https://img.shields.io/badge/Documentation-Open_LLM_VTuber-blue" alt="Documentation">
  </a>
  <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2" target="_blank">
    <img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow" alt="Roadmap">
  </a>
  <a href="https://trendshift.io/repositories/12358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/12358" alt="Trendshift">
  </a>
</p>

> **Open-LLM-VTuber** revolutionizes the way you interact with AI, offering a fully customizable and offline-capable virtual companion.

<details>
  <summary> 常见问题 Common Issues doc (Written in Chinese)</summary>
  <a href="https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq" target="_blank">https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq</a>
</details>

<details>
  <summary> User Survey</summary>
  <a href="https://forms.gle/w6Y6PiHTZr1nzbtWA" target="_blank">https://forms.gle/w6Y6PiHTZr1nzbtWA</a>
</details>

<details>
  <summary> 调查问卷(中文)</summary>
  <a href="https://wj.qq.com/s2/16150415/f50a/" target="_blank">https://wj.qq.com/s2/16150415/f50a/</a>
</details>

<br>

> :warning: This project is under active development.

> :warning:  Running the server remotely requires `https` configuration. See [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

## Key Features

*   **Cross-Platform Compatibility**: Works seamlessly on macOS, Linux, and Windows, supporting NVIDIA and non-NVIDIA GPUs, with options for CPU and cloud API use.
*   **Offline Mode**: Enjoy complete offline functionality using local models, ensuring privacy and data security.
*   **Web and Desktop Clients**: Offers both web and desktop clients, providing a rich interactive experience.  The desktop client features a unique transparent background "desktop pet" mode.
*   **Advanced Interaction**:
    *   Visual perception via camera, screen recording, and screenshots.
    *   Voice interruption to avoid AI hearing itself.
    *   Touch feedback for interacting with your AI companion.
    *   Live2D expression control.
    *   Pet Mode with transparent background, global top-most, and click-through.
    *   Display of AI's inner thoughts.
    *   AI proactive speaking.
    *   Persistent chat logs.
    *   TTS translation support.
*   **Extensive Model Support**:
    *   Large Language Models (LLMs): Ollama, OpenAI, Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, and more.
    *   Automatic Speech Recognition (ASR): sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   Text-to-Speech (TTS): sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Highly Customizable**:
    *   Simple configuration file modifications for module switching.
    *   Character customization: Import Live2D models, modify Prompts, and implement voice cloning.
    *   Flexible Agent implementation for integrating with any Agent architecture (HumeAI EVI, OpenAI Her, etc.).
    *   Modular design for easy integration of custom LLMs, ASR, TTS, and other features.

## Screenshots

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## Getting Started

Explore the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) section in our documentation.

## Update Instructions

Use `uv run update.py` to update the project if you've installed any versions after `v1.0.0`.

## Uninstall Instructions

Most files are in the project folder.  Models downloaded through ModelScope or Hugging Face may be in `MODELSCOPE_CACHE` or `HF_HOME`. Check the installation guide for removal of any extra tools like `uv`, `ffmpeg`, or `deeplx`.

## Contribute

Learn how to contribute to Open-LLM-VTuber by visiting the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) - Your Live2D desktop assistant powered by LLM.

## Third-Party Licenses

### Live2D Sample Models Notice

This project uses Live2D sample models under the [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and the [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html).  Commercial use may require additional licensing.

## Contributors

Thank you to our contributors!

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)

<hr>

[Back to Top](https://github.com/t41372/Open-LLM-VTuber)