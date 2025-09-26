<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

# Open-LLM-VTuber: Your AI Companion with a Live2D Avatar 

**Create your own interactive AI companion with voice, vision, and a dynamic Live2D avatar!** Dive into the world of AI-powered virtual interaction by exploring the [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) project.

<div align="center">
  <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber?style=for-the-badge" alt="GitHub release"/>
  <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber?style=for-the-badge" alt="License"/>
  <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg?style=for-the-badge" alt="CodeQL"/>
  <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg?style=for-the-badge" alt="Ruff"/>
  <img src="https://img.shields.io/docker/pulls/t41372/open-llm-vtuber?style=for-the-badge&logo=docker" alt="Docker Pulls"/>
  <a href="https://qm.qq.com/q/ngvNUQpuKI"><img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=for-the-badge&logo=qq&logoColor=white" alt="QQ Group"/></a>
  <a href="https://pd.qq.com/s/tt54r3bu"><img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=for-the-badge&logo=qq&logoColor=white" alt="QQ Channel"/></a>
  <a href="https://www.buymeacoffee.com/yi.ting"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"/></a>
  <a href="https://discord.gg/3UDA8YFDXx"><img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord Server"/></a>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"/></a>
  <a href="https://open-llm-vtuber.github.io/docs/quick-start"><img src="https://img.shields.io/badge/Documentation-Quick_Start-blue?style=for-the-badge" alt="Documentation"/></a>
  <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2"><img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow?style=for-the-badge" alt="Roadmap"/></a>
  <a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

[English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)

> **Note:** This project is under active development.

> **HTTPS Configuration for Remote Access:** If you plan to access the server remotely (e.g., on your phone) you will need to configure `https`.

## Key Features

*   **Cross-Platform Compatibility:** Supports Windows, macOS, and Linux.
*   **Offline Mode:** Operate completely offline, preserving your privacy and security.
*   **Versatile Clients:** Web and Desktop clients offer interactive features and customization. Includes a desktop pet mode.
*   **Advanced Interactions:**
    *   Visual perception (camera, screen recording, screenshots).
    *   Voice interruption without headphones.
    *   Touch feedback.
    *   Live2D expressions.
    *   Desktop pet mode with transparent background.
    *   AI thought display.
    *   AI proactive speaking.
    *   Persistent chat logs.
    *   TTS translation.
*   **Extensive Model Support:**
    *   **LLMs:** Ollama, OpenAI (and any OpenAI-compatible API), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
    *   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **High Customization:**
    *   Module configuration via config files.
    *   Custom Live2D models and persona.
    *   Agent architecture integration.
    *   Modular design for extensibility.

## Demos

|  |  |
|:---:|:---:|
|  |  |
|  |  |

## Quick Start

Get started by following the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) guide in the documentation.

## Update

Use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## Uninstall

Most files are stored in the project folder. Review the installation guide for additional tools you no longer need (e.g., `uv`, `ffmpeg`, `deeplx`).

## Contribute

Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to contribute.

## Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)

## Third-Party Licenses

### Live2D Sample Models Notice

This project uses Live2D sample models under the Live2D Free Material License Agreement and Terms of Use. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

## Contributors

Thanks our contributors and maintainers for making this project possible.

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)