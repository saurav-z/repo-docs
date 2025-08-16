<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

# Open-LLM-VTuber: Your AI Companion for Engaging Conversations and Visual Interaction

**Create your own interactive AI companion with voice and visual features, all running locally!** [Explore the project on GitHub](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

[![GitHub release](https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber)](https://github.com/t41372/Open-LLM-VTuber/releases)
[![license](https://img.shields.io/github/license/t41372/Open-LLM-VTuber)](https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE)
[![CodeQL](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml)
[![Ruff](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml)
[![Docker](https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue)](https://hub.docker.com/r/t41372/open-llm-vtuber)
[![QQ Group](https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white)](https://qm.qq.com/q/ngvNUQpuKI)
[![QQ Channel](https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white)](https://pd.qq.com/s/tt54r3bu)
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/yi.ting)
[![](https://dcbadge.limes.pink/api/server/3UDA8YFDXx)](https://discord.gg/3UDA8YFDXx)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber)

[English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)
[Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | [![Roadmap](https://img.shields.io/badge/Roadmap-GitHub_Project-yellow)](https://github.com/orgs/Open-LLM-VTuber/projects/2)
<a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

> Common Issues (Chinese): [https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq](https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq)
>
> User Survey: [https://forms.gle/w6Y6PiHTZr1nzbtWA](https://forms.gle/w6Y6PiHTZr1nzbtWA)
>
> 调查问卷(中文): [https://wj.qq.com/s2/16150415/f50a/](https://wj.qq.com/s2/16150415/f50a/)

> :warning: This project is under active development.

> :warning: To run the server remotely with a microphone, configure `https`. See [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux. Supports both NVIDIA and non-NVIDIA GPUs, and CPU fallback.
*   **Offline Mode:** Enjoy private conversations without needing an internet connection, using local models.
*   **Web and Desktop Clients:** Offers a web version and a feature-rich desktop client with a transparent background desktop pet mode.
*   **Advanced Interaction:**
    *   **Visual Perception:** AI can "see" via camera, screen recording, and screenshots.
    *   **Real-time Voice Interaction:** Voice interruption, eliminating echo.
    *   **Touch Feedback:** Interact through clicks and drags.
    *   **Live2D Expressions:** Model expressions based on backend emotion mapping.
    *   **Pet Mode:** Transparent background for your AI companion on your screen.
    *   **AI Thoughts:** See AI's internal state.
    *   **Proactive Speaking:** AI initiates conversations.
    *   **Chat Log Persistence:** Save and revisit past conversations.
    *   **TTS Translation:** Chat in one language, hear AI in another.
*   **Extensive Model Support:**  Integration with a wide range of LLMs (Ollama, OpenAI, Gemini, etc.), ASR engines (sherpa-onnx, Faster-Whisper, etc.), and TTS solutions (sherpa-onnx, Coqui-TTS, etc.).
*   **Highly Customizable:**
    *   **Module Configuration:** Easily swap components via configuration files.
    *   **Character Customization:** Import custom Live2D models and shape your AI's persona.
    *   **Agent Implementation:** Integrate with various Agent architectures.
    *   **Extensible Design:** Modular design lets you easily add new LLMs, ASR, TTS, and other components.

## Demo

| <img src="assets/i1.jpg" alt="Demo Image 1"> | <img src="assets/i2.jpg" alt="Demo Image 2"> |
|:---:|:---:|
| <img src="assets/i3.jpg" alt="Demo Image 3"> | <img src="assets/i4.jpg" alt="Demo Image 4"> |

## User Reviews

> Thanks to the developer for open-sourcing and sharing the girlfriend for everyone to use.
>
> This girlfriend has been used over 100,000 times.

## Quick Start

See our [Quick Start Guide](https://open-llm-vtuber.github.io/docs/quick-start) for installation instructions.

## Update

> :warning: `v1.0.0` has breaking changes. Update via `uv run update.py` if installed after `v1.0.0`. See [latest deployment guide](https://open-llm-vtuber.github.io/docs/quick-start) for pre-v1.0.0 users.

## Uninstall

Most files are in the project folder. Check `MODELSCOPE_CACHE` or `HF_HOME` for models and review the installation guide for extra tools like `uv`, `ffmpeg`, or `deeplx`.

## Contribute

Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) - Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

Thanks to our contributors and maintainers!

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)