<div align="center">
  <img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner" style="max-width: 100%;">
</div>

# Open-LLM-VTuber: Your AI Companion, Now with a Live2D Avatar!

> Experience the future of AI companionship with Open-LLM-VTuber, a fully customizable and offline-capable virtual AI companion with a Live2D avatar.

[<img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber?style=flat-square" alt="GitHub Release">](https://github.com/t41372/Open-LLM-VTuber/releases)
[<img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber?style=flat-square" alt="License">](https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE)
[<img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml)
[<img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff">](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml)
[<img src="https://img.shields.io/docker/pulls/t41372/open-llm-vtuber?style=flat-square&label=Docker%20Pulls" alt="Docker Pulls">](https://hub.docker.com/r/t41372/open-llm-vtuber)
[<img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Group">](https://qm.qq.com/q/ngvNUQpuKI)
[<img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Channel (dev)">](https://pd.qq.com/s/tt54r3bu)
[<img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat-square&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">](https://www.buymeacoffee.com/yi.ting)
[<img src="https://discord.com/invite/3UDA8YFDXx" alt="Discord Server">](https://discord.gg/3UDA8YFDXx)
[<img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">](https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber)

[English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)
[Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | [<img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow?style=flat-square" alt="Roadmap">](https://github.com/orgs/Open-LLM-VTuber/projects/2)
[<img src="https://trendshift.io/api/badge/repositories/12358" alt="Trendshift" style="width: 250px; height: 55px;">](https://trendshift.io/repositories/12358)

## Key Features of Open-LLM-VTuber

*   **Versatile AI Companion:** Create your virtual girlfriend, boyfriend, pet, or any character with a customizable Live2D avatar.
*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Offline Functionality:** Run entirely offline with local models for privacy and security.
*   **Web and Desktop Clients:** Enjoy a web interface and a desktop client with a unique transparent background desktop pet mode.
*   **Advanced Interactions:**
    *   Visual perception with camera, screen recording, and screenshots.
    *   Voice interruption without feedback.
    *   Touch feedback interaction.
    *   Live2D expression control.
    *   Desktop pet mode for on-screen companionship.
    *   Display AI's inner thoughts.
    *   AI proactive speaking.
    *   Persistent chat logs.
    *   TTS translation support.
*   **Broad Model Support:** Integrates with a wide array of Large Language Models, Automatic Speech Recognition, and Text-to-Speech solutions (see details below).
*   **Highly Customizable:**
    *   Simple module configuration.
    *   Character customization with custom Live2D models and persona modification.
    *   Flexible agent implementation.
    *   Excellent extensibility for adding new features.

## Why Choose Open-LLM-VTuber?

This project lets you create your AI companion that understands voice, sees you, and interacts with a lively Live2D avatar, all running on your computer.  Open-LLM-VTuber is ideal for anyone seeking an AI companion, from a virtual friend to a desktop pet, while ensuring privacy and control.

## Technical Details

### Backend Support:
*   **LLMs:** Ollama, OpenAI (and any OpenAI-compatible API), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
*   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
*   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.

## Getting Started

*   For detailed installation instructions, please consult the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) guide.

## Demo

| <img src="assets/i1.jpg" alt="Demo Image 1"> | <img src="assets/i2.jpg" alt="Demo Image 2"> |
|:---:|:---:|
| <img src="assets/i3.jpg" alt="Demo Image 3"> | <img src="assets/i4.jpg" alt="Demo Image 4"> |

## Updates
*   To update, use `uv run update.py`.

## Uninstall
*   Most files are in the project folder.  Also check for models in `MODELSCOPE_CACHE` or `HF_HOME`.

## Contribute

*   See the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) for contribution details.

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant): Your Live2D desktop assistant, available for Windows and MacOS.

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)

## Learn More

*   [GitHub Repository](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)