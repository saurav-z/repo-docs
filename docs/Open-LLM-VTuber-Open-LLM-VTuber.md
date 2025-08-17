<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion, Brought to Life!</h1>

<div align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber" target="_blank">
    <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="GitHub stars" />
  </a>
  <br />
  <a href="https://github.com/t41372/Open-LLM-VTuber/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber?style=flat-square" alt="GitHub release" />
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber?style=flat-square" alt="License" />
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml" target="_blank">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL" />
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml" target="_blank">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff" />
  </a>
  <a href="https://hub.docker.com/r/t41372/open-llm-vtuber" target="_blank">
    <img src="https://img.shields.io/docker/pulls/t41372/open-llm-vtuber?label=Docker%20Pulls&style=flat-square" alt="Docker Pulls" />
  </a>
    <a href="https://qm.qq.com/q/ngvNUQpuKI" target="_blank">
      <img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Group" />
    </a>
  <a href="https://pd.qq.com/s/tt54r3bu" target="_blank">
    <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Channel (dev)" />
  </a>
  <br/>
  <a href="https://www.buymeacoffee.com/yi.ting" target="_blank">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee" />
  </a>
  <a href="https://discord.gg/3UDA8YFDXx" target="_blank">
    <img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord Server" />
  </a>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" />
  </a>
  <br/>
  <a href="https://open-llm-vtuber.github.io/docs/quick-start" target="_blank">Documentation</a> | <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2" target="_blank">Roadmap</a>
  <br/>
   <a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

>  Create your own AI companion with Open-LLM-VTuber, a fully customizable, open-source project bringing AI-powered virtual avatars to life!

[English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)

*For Chinese users, please refer to the [Common Issues document](https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq) and the [User Survey](https://forms.gle/w6Y6PiHTZr1nzbtWA) (or its Chinese equivalent: [调查问卷](https://wj.qq.com/s2/16150415/f50a/)).*

<br/>
<div style="background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 10px; margin-bottom: 15px;">
  <p style="margin: 0;">:warning: This project is in its early stages of development and is actively being improved.  It is recommended to configure HTTPS if running the server remotely due to microphone access restrictions.</p>
</div>

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux, with GPU acceleration options.
*   **Offline Mode:** Enjoy privacy and security with complete offline functionality using local models.
*   **Web & Desktop Clients:** Choose between web and desktop versions with rich interaction and personalization.
*   **Advanced Interaction:**
    *   Visual perception with camera, screen recording, and screenshot support.
    *   Voice interruption and touch feedback for engaging interactions.
    *   Live2D expressions and a unique "Pet Mode" with transparent background.
    *   Display AI's inner thoughts, proactive AI speaking, and chat log persistence.
    *   TTS translation support.
*   **Extensive Model Support:**
    *   Large Language Models (LLMs): Ollama, OpenAI (and compatible APIs), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
    *   Automatic Speech Recognition (ASR): sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   Text-to-Speech (TTS): sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Highly Customizable:**
    *   Simple module configuration through configuration files.
    *   Character customization with custom Live2D models, prompts, and voice cloning.
    *   Flexible Agent implementation for integrating with various AI architectures.
    *   Modular design for easy extension with your own LLMs, ASR, and TTS implementations.

## Demo

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## Getting Started

Get up and running quickly with our [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) guide.

## Update Instructions

>  :warning: Version `v1.0.0` introduced breaking changes. If you have versions after `v1.0.0`, use `uv run update.py` to update.  For pre-v1.0.0 users, a fresh deployment using the [latest deployment guide](https://open-llm-vtuber.github.io/docs/quick-start) is recommended.

## Uninstall

Most files and models are stored within the project directory. Check the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`. Also, be sure to check `MODELSCOPE_CACHE` or `HF_HOME` for models downloaded via ModelScope or Hugging Face.

## Contribute

We welcome contributions! Please review our [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to learn how to get involved.

## Related Projects
*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant): Your LLM-powered Live2D desktop assistant!

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

Special thanks to our contributors!

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)