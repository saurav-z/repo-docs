<div align="center">
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber">
    <img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner" style="max-width: 100%;">
  </a>
  <h1>Open-LLM-VTuber: Your AI Companion, Powered by Open Source</h1>
  <p>Create a dynamic and engaging virtual companion with real-time voice interaction and a Live2D avatar, all running offline on your computer!</p>
  <p>
    <a href="https://github.com/t41372/Open-LLM-VTuber">
      <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="GitHub stars">
    </a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/releases">
      <img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber?style=flat-square" alt="GitHub release">
    </a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber?style=flat-square" alt="License">
    </a>
    <a href="https://hub.docker.com/r/t41372/open-llm-vtuber">
      <img src="https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker" alt="Docker">
    </a>
    <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber">
      <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
    </a>
    <a href="https://discord.gg/3UDA8YFDXx">
      <img src="https://img.shields.io/discord/1127983799661305355?label=Discord&logo=discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://qm.qq.com/q/ngvNUQpuKI">
      <img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Group">
    </a>
    <a href="https://pd.qq.com/s/tt54r3bu">
      <img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat-square&logo=qq&logoColor=white" alt="QQ Channel">
    </a>
     <a href="https://www.buymeacoffee.com/yi.ting">
      <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat-square&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee">
    </a>
    <a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
  </p>
  <p>
    <a href="https://open-llm-vtuber.github.io/docs/quick-start">Documentation</a> |
    <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2">Roadmap</a> |
    <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md">中文README</a>
  </p>
</div>

<hr>

> **Warning:** This project is under active development. Expect frequent updates and improvements!

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Offline Mode:** Enjoy privacy and security with complete offline functionality.
*   **Interactive Web & Desktop Clients:** Choose your preferred way to interact.  The desktop client offers unique "desktop pet" mode.
*   **Advanced Interaction:**
    *   Visual perception
    *   Voice interruption 
    *   Touch feedback
    *   Live2D expressions
    *   Pet Mode
    *   AI thoughts
    *   AI proactive speaking
    *   Chat log persistence
    *   TTS translation
*   **Extensive Model Support:**
    *   Large Language Models (LLMs): Supports a wide array of LLMs, including Ollama, OpenAI, Gemini, Claude, Mistral, and more.
    *   Automatic Speech Recognition (ASR): Integrates with sherpa-onnx, FunASR, Whisper.cpp, and others.
    *   Text-to-Speech (TTS): Compatible with sherpa-onnx, pyttsx3, Coqui-TTS, Edge TTS, and many other TTS engines.
*   **Highly Customizable:**
    *   Modular configuration
    *   Character customization
    *   Flexible Agent implementation
    *   Extensibility
    
## Demo

<div align="center">
    <img src="assets/i1.jpg" alt="Demo Image 1" width="30%">
    <img src="assets/i2.jpg" alt="Demo Image 2" width="30%">
    <img src="assets/i3.jpg" alt="Demo Image 3" width="30%">
    <img src="assets/i4.jpg" alt="Demo Image 4" width="30%">
</div>

## Getting Started

Find detailed instructions in the [Quick Start Guide](https://open-llm-vtuber.github.io/docs/quick-start) to get your AI companion up and running!

## Update

Use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## Uninstall

Most files, including Python dependencies and models, are stored in the project folder.
However, models downloaded via ModelScope or Hugging Face may also be in `MODELSCOPE_CACHE` or `HF_HOME`. While we aim to keep them in the project's `models` directory, it's good to double-check.  
Review the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`.  

## Contribute

We welcome contributions!  Check out the [Development Guide](https://docs.llmvtuber.com/docs/development-guide/overview) to learn how you can get involved.

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.


## Contributors

A big thank you to our contributors!

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)