[<img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner" />](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

# Open-LLM-VTuber: Create Your Own AI Companion 

**Bring your AI companion to life with real-time voice interaction and a stunning Live2D avatar, all running offline on your computer!** This innovative project, [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber), offers a unique and engaging way to interact with AI.

### Project Status & Resources

*   [GitHub Release](https://github.com/t41372/Open-LLM-VTuber/releases)
*   [License](https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE)
*   [CodeQL](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml)
*   [Ruff](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml)
*   [Docker](https://hub.docker.com/r/t41372/open-llm-vtuber)
*   [QQ Group](https://qm.qq.com/q/ngvNUQpuKI)
*   [QQ Channel (Dev)](https://pd.qq.com/s/tt54r3bu)
*   [Buy Me a Coffee](https://www.buymeacoffee.com/yi.ting)
*   [Discord](https://discord.gg/3UDA8YFDXx)
*   [Ask DeepWiki](https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber)
*   [English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)
*   [Documentation](https://open-llm-vtuber.github.io/docs/quick-start)
*   [Roadmap](https://github.com/orgs/Open-LLM-VTuber/projects/2)

<img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>

> Common Issues: [Chinese Documentation](https://docs.qq.com/pdf/DTFZGQXdTUXhIYWRq)
>
> User Survey: [Survey Form](https://forms.gle/w6Y6PiHTZr1nzbtWA)
>
> 调查问卷(中文): [Questionnaire](https://wj.qq.com/s2/16150415/f50a/)

> :warning: This project is in its early stages and is under active development.

> :warning: If you plan to run the server remotely, configure `https` due to microphone usage restrictions (see [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)).

## Key Features

*   **Cross-Platform Compatibility:** Seamlessly runs on Windows, macOS, and Linux. Supports NVIDIA and non-NVIDIA GPUs, with options for CPU or cloud-based processing.
*   **Offline Mode:** Enjoy complete privacy and security with full offline functionality using local models.
*   **Versatile Clients:**  Access your AI companion through both web and desktop clients, offering rich interactivity and customization. Includes a unique desktop pet mode.
*   **Advanced Interaction:**
    *   **Visual Perception:**  AI companion can see you and your screen.
    *   **Hands-Free Voice Interruption:**  AI won't hear its own voice.
    *   **Touch Interaction:**  Click and drag to interact.
    *   **Live2D Expressions:** Control your AI companion's expressions.
    *   **Desktop Pet Mode:**  Transparent background for a floating companion.
    *   **Inner Thoughts Display:** See the AI's expressions and thoughts.
    *   **Proactive Speaking:** AI can initiate conversations.
    *   **Chat Log Persistence:**  Resume previous conversations.
    *   **TTS Translation:**  Chat in one language, while the AI speaks in another.
*   **Extensive Model Support:**
    *   **LLMs:** Ollama, OpenAI (and compatible APIs), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, and more.
    *   **ASR:** sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, and more.
    *   **TTS:** sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, and more.
*   **Highly Customizable:**
    *   **Module Configuration:** Easily switch between different functionalities.
    *   **Character Customization:** Import custom Live2D models, modify prompts, and clone voices.
    *   **Agent Integration:** Implement Agent architectures like HumeAI EVI and OpenAI Her.
    *   **Extensibility:**  Add your own LLMs, ASR, TTS, and other modules.

## Demo

<p align="center">
  <img src="./assets/i1.jpg" width="30%" alt="Demo Image 1">
  <img src="./assets/i2.jpg" width="30%" alt="Demo Image 2">
  <img src="./assets/i3.jpg" width="30%" alt="Demo Image 3">
  <img src="./assets/i4.jpg" width="30%" alt="Demo Image 4">
</p>

## User Feedback

> "Thanks to the developer for open-sourcing and sharing the girlfriend for everyone to use."
>
> "This girlfriend has been used over 100,000 times."

## Quick Start

See the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) section in our documentation.

## Update

> :warning:  Version `v1.0.0` has breaking changes. Upgrade via `uv run update.py` (for versions after `v1.0.0`). If coming from earlier versions, redeploy following the [latest deployment guide](https://open-llm-vtuber.github.io/docs/quick-start).

## Uninstall

Most files are stored in the project folder. Review the installation guide for unnecessary tools like `uv`, `ffmpeg`, and `deeplx`.

## Contribute

Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant): A Live2D desktop assistant powered by LLM, supporting screen sensing, voice commands, and more.

## Third-Party Licenses

### Live2D Sample Models Notice

This project uses Live2D sample models under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data.

Note: Commercial use may require additional licensing from Live2D Inc. if using these sample models.

## Contributors

[Contributors](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)