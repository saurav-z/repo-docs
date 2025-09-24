<div align="center">
  <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber">
    <img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner" style="width: 100%; max-width: 800px;">
  </a>
  <h1>Open-LLM-VTuber</h1>
  <p><b>Bring your AI companion to life with real-time voice interaction and a captivating Live2D avatar, all running offline on your computer!</b></p>
</div>

[![GitHub release](https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber)](https://github.com/t41372/Open-LLM-VTuber/releases) 
[![license](https://img.shields.io/github/license/t41372/Open-LLM-VTuber)](https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE) 
[![CodeQL](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml)
[![Ruff](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml)
[![Docker](https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue)](https://hub.docker.com/r/t41372/open-llm-vtuber) 
[![QQ Group](https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white)](https://qm.qq.com/q/ngvNUQpuKI)
[![QQ Channel](https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white)](https://pd.qq.com/s/tt54r3bu)
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/yi.ting)
[![Discord Server](https://dcbadge.limes.pink/api/server/3UDA8YFDXx)](https://discord.gg/3UDA8YFDXx)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber)

[English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)
[Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | [![Roadmap](https://img.shields.io/badge/Roadmap-GitHub_Project-yellow)](https://github.com/orgs/Open-LLM-VTuber/projects/2)
<a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

---

## 🔑 Key Features of Open-LLM-VTuber

*   **Cross-Platform Compatibility:** Seamlessly runs on Windows, macOS, and Linux, supporting both NVIDIA and non-NVIDIA GPUs, and CPU fallback.
*   **Offline Mode:**  Enjoy complete privacy and security with fully offline functionality, using local models without an internet connection.
*   **Versatile Clients:** Experience the AI companion through a web interface or a feature-rich desktop client.
*   **Advanced Interaction:**
    *   Visual Perception (Camera, Screen Recording, Screenshots)
    *   Voice Interruption (AI doesn't hear itself)
    *   Touch Feedback
    *   Live2D Expressions
    *   Desktop Pet Mode
    *   AI Inner Thoughts Display
    *   Proactive AI Speaking
    *   Persistent Chat Logs
    *   TTS Translation Support
*   **Extensive Model Support:** Wide range of Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) options.
*   **Highly Customizable:**
    *   Simple Module Configuration
    *   Character Customization (Live2D, Prompts, Voice Cloning)
    *   Flexible Agent Implementation
    *   Extensible Design for easy module additions (LLMs, ASR, TTS)

## 💻 What is Open-LLM-VTuber?

Open-LLM-VTuber is a cutting-edge open-source project that brings your AI companion to life.  It allows you to create a virtual character capable of real-time voice conversations, visual interaction, and expressive movement through a Live2D avatar.  It is designed to be your AI girlfriend, boyfriend, pet, or any character that fits your imagination.  Powered by a variety of LLMs, ASR, and TTS solutions, all running locally on your computer.  

This project aims to replicate the functionality of the closed-source AI Vtuber "neuro-sama," using open-source solutions that can run offline.

## 🖼️ Demo

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## 🚀 Get Started

For detailed installation instructions, consult the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) section of the documentation.

## ⚠️ Important Notices

*   This project is under active development.
*   `v1.0.0` introduced breaking changes requiring re-deployment and potential dependency reinstallation.
*   For remote server access via HTTPS, see the [MDN Web Doc](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

## 🔧 Uninstalling

Most project files, including Python dependencies and models, are stored within the project folder. Model downloads from ModelScope or Hugging Face may also be located in `MODELSCOPE_CACHE` or `HF_HOME`.  Refer to the installation guide for any required extra tools like `uv`, `ffmpeg`, or `deeplx` if these are no longer needed.

## 🙏 Contributing

We welcome contributions!  Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to get started.

---

## 🎉 Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) - Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

---

## 📜 Third-Party Licenses

### Live2D Sample Models Notice

This project utilizes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

---

## ✨ Acknowledgements
Thanks our contributors and maintainers for making this project possible.

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)