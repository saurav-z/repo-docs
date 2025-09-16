<div align="center">
  <img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner" width="100%">
  <h1>Open-LLM-VTuber: Your AI Companion, Brought to Life</h1>
  <p>Transform your desktop with a voice-interactive AI companion featuring a Live2D avatar – all running locally! <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber">Explore the project on GitHub</a>.</p>
</div>

---

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

English README | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)

[Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | [![Roadmap](https://img.shields.io/badge/Roadmap-GitHub_Project-yellow)](https://github.com/orgs/Open-LLM-VTuber/projects/2)

<a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

---
> **Note:** This project is under active development.

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Offline Mode:**  Operate entirely offline using local models for privacy and security.
*   **Interactive Web & Desktop Clients:**  Web and desktop clients with interactive features and customization options. Desktop pet mode is supported.
*   **Advanced Interaction:**
    *   Visual perception via camera, screen recording, and screenshots.
    *   Voice interruption and touch feedback.
    *   Live2D expression control.
    *   Pet mode with a transparent background.
    *   Display AI's inner thoughts, proactive speaking, and chat log persistence.
    *   TTS Translation Support
*   **Extensive Model Support:**
    *   **LLMs:** Ollama, OpenAI (and any OpenAI-compatible API), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
    *   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
    *   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **High Customization:**
    *   Simple module configuration.
    *   Character customization with Live2D models, prompts, and voice cloning.
    *   Flexible Agent implementation with Agent interface integration.
    *   Modular design for easy extensibility of LLMs, ASR, TTS, and other features.

---

## Demos

*   [Demo 1](assets/i1.jpg)
*   [Demo 2](assets/i2.jpg)
*   [Demo 3](assets/i3.jpg)
*   [Demo 4](assets/i4.jpg)

---

## Quick Start

Get up and running quickly with our detailed [Quick Start Guide](https://open-llm-vtuber.github.io/docs/quick-start).

---

## Additional Resources

*   [Common Issues (Chinese)](https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq)
*   [User Survey](https://forms.gle/w6Y6PiHTZr1nzbtWA)
*   [调查问卷 (Chinese)](https://wj.qq.com/s2/16150415/f50a/)

---

## Updating and Uninstalling

*   **Update:** Use `uv run update.py` to update your installation.
*   **Uninstall:**  Most files are within the project folder, but check `MODELSCOPE_CACHE` or `HF_HOME` for downloaded models.

---

## Contributing

We welcome your contributions!  See the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) for details.

---

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant): Your Live2D desktop assistant powered by LLM!

---

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

---

## Contributors

[Contributor Images](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)