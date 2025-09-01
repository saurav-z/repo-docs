<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion with a Live2D Avatar</h1>

<h3 align="center">
  [GitHub Repository](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) | 
  [Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | 
  [Roadmap](https://github.com/orgs/Open-LLM-VTuber/projects/2)
</h3>

[![GitHub release](https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber)](https://github.com/t41372/Open-LLM-VTuber/releases)
[![License](https://img.shields.io/github/license/t41372/Open-LLM-VTuber)](https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE)
[![CodeQL](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml)
[![Ruff](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml)
[![Docker](https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue)](https://hub.docker.com/r/t41372/open-llm-vtuber)
[![QQ Group](https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white)](https://qm.qq.com/q/ngvNUQpuKI)
[![QQ Channel](https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white)](https://pd.qq.com/s/tt54r3bu)
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/yi.ting)
[![](https://dcbadge.limes.pink/api/server/3UDA8YFDXx)](https://discord.gg/3UDA8YFDXx)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber)

English README | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)
<a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>


> :warning: This project is under **active development**.

> :warning: To access the server remotely, configure `https` due to the microphone requirement on the front end.

## About Open-LLM-VTuber

**Transform your desktop with Open-LLM-VTuber, a powerful, offline-capable AI companion featuring a captivating Live2D avatar, ready to converse with you in real-time!** This project offers a unique blend of AI interaction and visual engagement, allowing you to create your ideal virtual companion. Whether you envision a virtual friend, a cute pet, or a dynamic character, Open-LLM-VTuber provides the tools and flexibility to bring your vision to life. With comprehensive support for Windows, macOS, and Linux, alongside both web and desktop client options, you can experience seamless interaction with your AI companion, including a special **transparent background desktop pet mode** that lets your AI avatar accompany you anywhere on your screen.  Thanks to chat log persistence, you can continue your conversations without interruption.

### Key Features:

*   **Cross-Platform Compatibility:** Fully supports Windows, macOS, and Linux.
*   **Offline Mode:** Utilize local models for privacy and accessibility, with no internet required.
*   **Versatile Clients:** Experience AI companionship through both web and desktop applications, including a transparent background desktop pet mode.
*   **Advanced Interactions:**
    *   Visual perception via camera and screen recording
    *   Voice interruption without feedback
    *   Touch feedback
    *   Live2D expressions
    *   Pet mode with transparency
    *   AI inner thought display
    *   Proactive speaking feature
    *   Chat log persistence
    *   TTS Translation
*   **Extensive Model Support:** Compatible with a wide array of Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) engines, including:
    *   LLMs: Ollama, OpenAI, Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
    *   ASR: sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   TTS: sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Highly Customizable:**
    *   Simple module configuration.
    *   Character customization with Live2D models, prompts, and voice cloning.
    *   Flexible Agent integration with support for HumeAI EVI, OpenAI Her, Mem0, and more.
    *   Modular design for easy expansion with new LLMs, ASR, and TTS modules.

### Demo
| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## User Reviews
> Thanks to the developer for open-sourcing and sharing the girlfriend for everyone to use
> 
> This girlfriend has been used over 100,000 times

## Quick Start

Get started with Open-LLM-VTuber!  Refer to the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) section in our documentation for detailed installation instructions.

## Update
Use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## Uninstall

Most files, including Python dependencies and models, are stored in the project folder.

However, models downloaded via ModelScope or Hugging Face may also be in `MODELSCOPE_CACHE` or `HF_HOME`. While we aim to keep them in the project's `models` directory, it's good to double-check.  

Review the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`.

## Contribute

We welcome contributions! Please see the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to get started.

## 🎉 Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)
- Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

## 📜 Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors
Thanks our contributors and maintainers for making this project possible.

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)