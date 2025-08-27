<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion, Powered by Voice and Vision</h1>

<h3 align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber" target="_blank">
    <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="GitHub stars" />
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber" target="_blank">
    <img src="https://img.shields.io/github/release/t41372/Open-LLM-VTuber?style=flat-square" alt="GitHub release" />
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber?style=flat-square" alt="License" />
  </a>
    <a href="https://hub.docker.com/r/t41372/open-llm-vtuber" target="_blank">
    <img src="https://img.shields.io/docker/pulls/t41372/open-llm-vtuber?style=flat-square&logo=docker" alt="Docker Pulls" />
  </a>
  <a href="https://discord.gg/3UDA8YFDXx" target="_blank">
    <img src="https://img.shields.io/discord/1180342357687068702?label=Discord&logo=discord&style=flat-square" alt="Discord" />
  </a>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="DeepWiki" />
  </a>
</h3>

**Open-LLM-VTuber** is a cutting-edge project that brings your AI companion to life with voice interaction, visual perception, and a captivating Live2D avatar, all running locally on your computer.

[Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | [Roadmap](https://github.com/orgs/Open-LLM-VTuber/projects/2) | [Original Repo](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Offline Mode:** Enjoy complete privacy and security with local model processing, no internet required.
*   **Web and Desktop Clients:** Utilize both web and desktop versions for versatile access and interaction.
*   **Advanced Interaction:**
    *   **Visual Perception:** AI can "see" via camera, screen recording, and screenshots.
    *   **Real-time Voice Interruption:** Experience natural conversations without echo.
    *   **Touch Feedback:** Interact with your companion through clicks and drags.
    *   **Expressive Live2D Avatars:** Control your avatar's expressions dynamically.
    *   **Desktop Pet Mode:** Have your AI companion accompany you anywhere on your screen.
    *   **AI Thought Display:** See AI's internal thoughts and actions.
    *   **Proactive AI Speaking:**  AI can initiate conversation.
    *   **Persistent Chat Logs:** Never lose your conversations, switch back to previous chats anytime.
    *   **Translation Support:** Communicate in your preferred language with AI voice in another.
*   **Extensive Model Support:**
    *   **LLMs:** Ollama, OpenAI, Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, and more.
    *   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Highly Customizable:**
    *   **Modular Configuration:** Easily configure modules through simple file edits.
    *   **Character Customization:** Import custom Live2D models and create unique personas.
    *   **Agent Integration:** Implement Agent architectures such as HumeAI EVI, OpenAI Her, and Mem0.
    *   **Extensible Architecture:** Add LLMs, ASR, TTS, and other modules.

## Screenshots

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## Quick Start

Get up and running with Open-LLM-VTuber quickly by following the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) instructions in our documentation.

## Updates

If you installed versions after v1.0.0, use `uv run update.py` to update.

## Uninstalling

Most files are stored in the project directory. Also check `MODELSCOPE_CACHE` or `HF_HOME` for downloaded models. Review the installation guide for any extra tools like `uv`, `ffmpeg`, or `deeplx`.

## Contributing

Interested in contributing?  Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## Third-Party Licenses

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