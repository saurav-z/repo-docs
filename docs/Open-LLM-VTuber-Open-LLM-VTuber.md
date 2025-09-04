[![](./assets/banner.jpg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

# Open-LLM-VTuber: Your AI Companion in a Live2D Avatar 

**Bring your AI companion to life with Open-LLM-VTuber, a versatile, open-source project that lets you create and interact with a customizable AI-powered Live2D character – all running locally!** [Explore the project on GitHub](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber).

<div align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber/releases">
    <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber" alt="GitHub release" />
  </a>
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber" alt="license" />
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL" />
  </a>
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml">
    <img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff" />
  </a>
  <a href="https://hub.docker.com/r/t41372/open-llm-vtuber">
    <img src="https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue" alt="Docker" />
  </a>
  <a href="https://qm.qq.com/q/ngvNUQpuKI">
    <img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white" alt="QQ Group" />
  </a>
  <a href="https://pd.qq.com/s/tt54r3bu">
    <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white" alt="QQ Channel" />
  </a>
  <a href="https://www.buymeacoffee.com/yi.ting">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="BuyMeACoffee" />
  </a>
  <a href="https://discord.gg/3UDA8YFDXx">
    <img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord" />
  </a>
  <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" />
  </a>

  <a href="https://open-llm-vtuber.github.io/docs/quick-start">Documentation</a> |
  <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2">Roadmap</a>
</div>

<div align="center">
  <a href="https://trendshift.io/repositories/12358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

> **Quick Links:**
>
> *   Common Issues (Chinese): [https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq](https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq)
> *   User Survey: [https://forms.gle/w6Y6PiHTZr1nzbtWA](https://forms.gle/w6Y6PiHTZr1nzbtWA)
> *   Survey (Chinese): [https://wj.qq.com/s2/16150415/f50a/](https://wj.qq.com/s2/16150415/f50a/)

> :warning: This project is under active development, with new features being added regularly.

> :warning:  For remote access (e.g., accessing the server on your phone), you'll need to configure HTTPS due to the microphone's requirement for a secure context.  Use a reverse proxy for this purpose.  See [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) for details.

## Key Features

*   **Cross-Platform:**  Works seamlessly on Windows, macOS, and Linux, with options for NVIDIA and non-NVIDIA GPUs and CPU fallback.
*   **Offline Mode:**  Enjoy complete privacy and security with fully offline operation using local models.
*   **Web & Desktop Clients:** Use either a web browser or a feature-rich desktop client for versatile interaction.  The desktop client offers a "desktop pet" mode with a transparent background.
*   **Advanced Interactions:**
    *   Visual perception with camera, screen recording, and screenshot support.
    *   Voice interruption without hearing its own voice
    *   Touch feedback.
    *   Live2D expression control.
    *   "Pet mode" with transparent background and click-through functionality.
    *   AI's inner thoughts and feelings display.
    *   AI proactive speaking feature.
    *   Chat log persistence for continuous conversations.
    *   TTS translation support.
*   **Extensive Model Support:**
    *   **LLMs:** Ollama, OpenAI (and API-compatible services), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, and more.
    *   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **High Customization:**
    *   Simple module configuration via configuration files.
    *   Character customization with custom Live2D models, prompt modification, and voice cloning.
    *   Flexible Agent implementation for integrating various Agent architectures.
    *   Extensibility for easy addition of new LLM, ASR, TTS, and other modules.

## Demo
<div align="center">
  <img src="./assets/i1.jpg" width="30%" hspace="4px"> <img src="./assets/i2.jpg" width="30%" hspace="4px">
  <br>
  <img src="./assets/i3.jpg" width="30%" hspace="4px"> <img src="./assets/i4.jpg" width="30%" hspace="4px">
</div>

## Get Started

Find quick start instructions and detailed documentation in the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) section of our documentation.

## Update and Uninstall

*   **Update:** Use `uv run update.py` to update after installation.
*   **Uninstall:** Most project files are within the main directory.  Check the installation guide for additional tools to remove (e.g., `uv`, `ffmpeg`).

## Contribute

See the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to start contributing!

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) - Your Live2D desktop assistant powered by LLM!

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

Thank you to our contributors!

[![Contributors](https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)