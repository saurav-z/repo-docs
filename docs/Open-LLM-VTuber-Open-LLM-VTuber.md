<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion, Brought to Life</h1>
<h3 align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber">
    <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber">
    <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber" alt="GitHub release">
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber">
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
    <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white" alt="QQ Channel (dev)">
  </a>
  <a href="https://www.buymeacoffee.com/yi.ting">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee">
  </a>
  <a href="https://discord.gg/3UDA8YFDXx">
    <img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord">
  </a>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md">
    <img src="https://img.shields.io/badge/Read%20in-中文-blue" alt="中文README">
  </a>
  <a href="https://open-llm-vtuber.github.io/docs/quick-start">
    <img src="https://img.shields.io/badge/Documentation-online-green" alt="Documentation">
  </a>
  <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2">
    <img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow" alt="Roadmap">
  </a>
  <a href="https://trendshift.io/repositories/12358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55">
  </a>
</h3>


> **Looking for a real-time, voice-interactive AI companion with a captivating Live2D avatar?**  Open-LLM-VTuber lets you create a personalized virtual friend that lives on your computer!

> 常见问题 Common Issues doc (Written in Chinese): https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq
>
> User Survey: https://forms.gle/w6Y6PiHTZr1nzbtWA
>
> 调查问卷(中文): https://wj.qq.com/s2/16150415/f50a/

> :warning: This project is in its early stages and is currently under **active development**.

> :warning: If you want to run the server remotely and access it on a different machine, such as running the server on your computer and access it on your phone, you will need to configure `https`, because the microphone on the front end will only launch in a secure context (a.k.a. https or localhost). See [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia). Therefore, you should configure https with a reverse proxy to access the page on a remote machine (non-localhost).

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux, with options for NVIDIA and non-NVIDIA GPUs, as well as CPU and cloud API support. Some components have GPU acceleration on macOS.
*   **Offline Mode:** Enjoy private and secure conversations with full offline functionality using local models; no internet connection needed.
*   **Web and Desktop Clients:** Choose between a user-friendly web interface or a powerful desktop client, featuring special **transparent background desktop pet mode** for constant companionship.
*   **Advanced Interaction:**
    *   **Visual Perception:** Uses camera, screen recording, and screenshots for your AI to see you and your screen.
    *   **Voice Interaction:** Voice interruption without headphones.
    *   **Touch Feedback:** Interact with your AI companion through clicks or drags.
    *   **Live2D Expressions:** Set emotion mapping to control model expressions from the backend.
    *   **Pet Mode:** Transparent background, global top-most, and mouse click-through.
    *   **AI Thoughts:** Display AI's inner thoughts.
    *   **Proactive Speaking:** AI proactive speaking feature.
    *   **Chat Log Persistence:** Switch to previous conversations anytime.
    *   **TTS Translation Support:** (e.g., chat in Chinese while AI uses Japanese voice)
*   **Extensive Model Support:**
    *   **LLMs:** Ollama, OpenAI (and any OpenAI-compatible API), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
    *   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Highly Customizable:**
    *   **Simple Module Configuration:** Modify functional modules through easy configuration file changes, without complex code modifications.
    *   **Character Customization:** Import custom Live2D models, modify prompts, and use voice cloning to create a unique AI companion.
    *   **Flexible Agent Implementation:** Integrate any Agent architecture via the Agent interface (HumeAI EVI, OpenAI Her, Mem0, etc.)
    *   **Extensibility:** Modular design allows easy addition of LLM, ASR, TTS, and other modules, expanding functionality at any time.

## Demo

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## User Reviews

> Thanks to the developer for open-sourcing and sharing the girlfriend for everyone to use
>
> This girlfriend has been used over 100,000 times

## Quick Start

Get started with Open-LLM-VTuber by following our comprehensive [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) guide in our documentation.

## Updates

> :warning: `v1.0.0` has breaking changes and requires re-deployment. You *may* still update via the method below, but the `conf.yaml` file is incompatible and most of the dependencies needs to be reinstalled with `uv`. For those who came from versions before `v1.0.0`, I recommend deploy this project again with the [latest deployment guide](https://open-llm-vtuber.github.io/docs/quick-start).

Use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## Uninstall

Most files, including Python dependencies and models, are stored in the project folder.

However, models downloaded via ModelScope or Hugging Face may also be in `MODELSCOPE_CACHE` or `HF_HOME`. While we aim to keep them in the project's `models` directory, it's good to double-check.

Review the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`.

## Contribute

Help shape the future of Open-LLM-VTuber!  Check out our [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to learn how you can contribute.

## Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)
- Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

A huge thank you to all our contributors!  See the full list on our [contributors page](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors).

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)