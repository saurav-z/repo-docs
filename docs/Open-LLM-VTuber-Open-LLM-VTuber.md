# Open-LLM-VTuber: Your Offline, Interactive AI Companion (Voice & Live2D)

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

<br>

Open-LLM-VTuber brings your AI companion to life with real-time voice interaction, a Live2D avatar, and fully offline functionality.

[**Visit the GitHub Repository for more information and to get started!**](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

## Key Features:

*   **Offline-First:** Run entirely offline with local models, ensuring privacy and no internet dependency.
*   **Cross-Platform:** Compatible with Windows, macOS, and Linux.
*   **Voice-Interactive:** Engage in real-time voice conversations with your AI companion.
*   **Live2D Avatar:** Enjoy a dynamic and expressive Live2D avatar.
*   **Customization:** Tailor your AI companion's appearance and personality.
*   **Desktop & Web Clients:** Offers both web version and desktop client usage modes, supporting rich interactive features and personalization settings. The desktop client can switch freely between window mode and desktop pet mode, allowing the AI companion to be by your side at all times.
*   **Advanced Interaction Features:**
  -   👁️ Visual perception, supporting camera, screen recording and screenshots, allowing your AI companion to see you and your screen
  -   🎤 Voice interruption without headphones (AI won't hear its own voice)
  -   🫱 Touch feedback, interact with your AI companion through clicks or drags
  -   😊 Live2D expressions, set emotion mapping to control model expressions from the backend
  -   🐱 Pet mode, supporting transparent background, global top-most, and mouse click-through - drag your AI companion anywhere on the screen
  -   💭 Display AI's inner thoughts, allowing you to see AI's expressions, thoughts and actions without them being spoken
  -   🗣️ AI proactive speaking feature
  -   💾 Chat log persistence, switch to previous conversations anytime
  -   🌍 TTS translation support (e.g., chat in Chinese while AI uses Japanese voice)
*   **Extensive Model Support:**
  -   🤖 Large Language Models (LLM): Ollama, OpenAI (and any OpenAI-compatible API), Gemini, Claude, Mistral, DeepSeek, Zhipu AI, GGUF, LM Studio, vLLM, etc.
  -   🎙️ Automatic Speech Recognition (ASR): sherpa-onnx, FunASR, Faster-Whisper, Whisper.cpp, Whisper, Groq Whisper, Azure ASR, etc.
  -   🔊 Text-to-Speech (TTS): sherpa-onnx, pyttsx3, MeloTTS, Coqui-TTS, GPTSoVITS, Bark, CosyVoice, Edge TTS, Fish Audio, Azure TTS, etc.
*   **Highly Customizable**:
  -   ⚙️ Simple module configuration: Switch various functional modules through simple configuration file modifications, without delving into the code
  -   🎨 Character customization: Import custom Live2D models to give your AI companion a unique appearance. Shape your AI companion's persona by modifying the Prompt. Perform voice cloning to give your AI companion the voice you desire
  -   🧩 Flexible Agent implementation: Inherit and implement the Agent interface to integrate any Agent architecture, such as HumeAI EVI, OpenAI Her, Mem0, etc.
  -   🔌 Good extensibility: Modular design allows you to easily add your own LLM, ASR, TTS, and other module implementations, extending new features at any time

## Demo

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## Quick Start

Refer to the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) section in our documentation for installation instructions.

## Contribution

Checkout the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

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