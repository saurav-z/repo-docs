[![](./assets/banner.jpg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

# Open-LLM-VTuber: Create Your AI Companion!

> Bring a charming, interactive AI companion to life with **Open-LLM-VTuber**, a fully customizable, open-source project that combines LLMs, Live2D avatars, and voice interaction, all running locally on your computer!

**[View the Project on GitHub](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)**

<div align="center">
    <a href="https://github.com/t41372/Open-LLM-VTuber/releases"><img alt="GitHub release" src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber"/></a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE"><img alt="license" src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber"/></a>
    <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg"/></a>
    <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml"><img alt="Ruff" src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg"/></a>
    <a href="https://hub.docker.com/r/t41372/open-llm-vtuber"><img alt="Docker" src="https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue"/></a>
    <a href="https://qm.qq.com/q/ngvNUQpuKI"><img alt="QQ Group" src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white"/></a>
    <a href="https://pd.qq.com/s/tt54r3bu"><img alt="QQ Channel (dev)" src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white"/></a>
    <a href="https://www.buymeacoffee.com/yi.ting"><img alt="BuyMeACoffee" src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black"/></a>
    <a href="https://discord.gg/3UDA8YFDXx"><img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx"/></a>
    <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"/></a>

    <a href="https://open-llm-vtuber.github.io/docs/quick-start">Documentation</a> | 
    <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2">Roadmap</a>
    
    <a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

>  Common Issues (Chinese): [https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq](https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq)
>  User Survey: [https://forms.gle/w6Y6PiHTZr1nzbtWA](https://forms.gle/w6Y6PiHTZr1nzbtWA)
>  调查问卷(中文): [https://wj.qq.com/s2/16150415/f50a/](https://wj.qq.com/s2/16150415/f50a/)

<div style="color:orange">
:warning: This project is under active development and in its early stages.
</div>

<div style="color:orange">
:warning: To access the server remotely (e.g., from your phone), configure `https`.  Microphone access requires a secure context (HTTPS or localhost). See [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) for more information. Use a reverse proxy for HTTPS.
</div>

## Key Features

*   **Cross-Platform Compatibility:** Runs seamlessly on Windows, macOS, and Linux. Includes support for NVIDIA and non-NVIDIA GPUs, with CPU fallback.
*   **Offline Mode:**  Enjoy complete privacy and security with local model support.  No internet connection is required for core functionality.
*   **Versatile Clients:**  Use the web version or the feature-rich desktop client (with a desktop pet mode!)
*   **Interactive Capabilities:**
    *   Visual Perception:  Utilizes your camera, screen recording, and screenshots.
    *   Voice Interruption:  The AI companion won't hear itself.
    *   Touch Feedback: Interact using clicks or drags.
    *   Live2D Expressions: Control model expressions.
    *   Pet Mode: Enjoy the AI as a transparent, always-on companion.
    *   Inner Thoughts: View the AI's thoughts, expressions, and actions.
    *   Proactive Speaking: AI can initiate conversations.
    *   Chat Log Persistence: Save and revisit past conversations.
    *   TTS Translation: Chat in one language, have the AI respond in another!
*   **Extensive Model Support:** Compatible with a wide variety of LLMs, ASRs, and TTS engines.
    *   LLMs: Ollama, OpenAI, Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, and more.
    *   ASR: sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   TTS: sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Customization Options:**
    *   Module Configuration: Easy configuration through simple file changes.
    *   Character Customization: Import Live2D models, customize prompts, and clone voices.
    *   Agent Integration: Implement your own agent architectures.
    *   Extensibility: Modular design allows easy addition of new modules.

## Demo

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## Quick Start

Get started with Open-LLM-VTuber by following the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) guide in our documentation.

## Update Instructions

<div style="color:orange">
:warning: `v1.0.0` introduced breaking changes. If you're updating from a previous version,  re-deployment is necessary.
</div>

To update after `v1.0.0`, use `uv run update.py`.

## Uninstall

Most project files, dependencies, and models are stored in the project folder.  However, models downloaded through ModelScope or Hugging Face might also be found in `MODELSCOPE_CACHE` or `HF_HOME`. Double-check the `models` directory.

Refer to the installation guide for cleaning up tools like `uv`, `ffmpeg`, or `deeplx`.

## Contribute

Contribute to Open-LLM-VTuber!  Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)
- Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

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