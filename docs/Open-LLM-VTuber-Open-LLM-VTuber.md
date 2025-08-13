<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/banner.jpg">
  <img alt="Open-LLM-VTuber Banner" src="./assets/banner.jpg">
</picture>

<h1 align="center">Open-LLM-VTuber: Your AI Companion, Now in Live2D!</h1>

<p align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber">
    <img src="https://img.shields.io/github/stars/t41372/Open-LLM-VTuber?style=social" alt="GitHub stars">
  </a>
</p>

<p align="center">
  Bring a dynamic AI companion to life with voice interaction, visual perception, and a customizable Live2D avatar—all running locally on your computer!
</p>

<p align="center">
    <a href="https://github.com/t41372/Open-LLM-VTuber/releases"><img src="https://img.shields.io/github/v/release/t41372/Open-LLM-VTuber" alt="GitHub release"></a>
    <a href="https://github.com/t41372/Open-LLM-VTuber/blob/master/LICENSE"><img src="https://img.shields.io/github/license/t41372/Open-LLM-VTuber" alt="license"></a>
    <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml"><img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
    <a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml"><img src="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/actions/workflows/ruff.yml/badge.svg" alt="Ruff"></a>
    <a href="https://hub.docker.com/r/t41372/open-llm-vtuber"><img src="https://img.shields.io/badge/t41372%2FOpen--LLM--VTuber-%25230db7ed.svg?logo=docker&logoColor=blue&labelColor=white&color=blue" alt="Docker"></a>
    <a href="https://qm.qq.com/q/ngvNUQpuKI"><img src="https://img.shields.io/badge/QQ_Group-792615362-white?style=flat&logo=qq&logoColor=white" alt="QQ Group"></a>
    <a href="https://pd.qq.com/s/tt54r3bu"><img src="https://img.shields.io/badge/QQ_Channel_(dev)-pd93364606-white?style=flat&logo=qq&logoColor=white" alt="QQ Channel (dev)"></a>
    <a href="https://www.buymeacoffee.com/yi.ting"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
    <a href="https://discord.gg/3UDA8YFDXx"><img src="https://dcbadge.limes.pink/api/server/3UDA8YFDXx" alt="Discord"></a>
    <a href="https://deepwiki.com/Open-LLM-VTuber/Open-LLM-VTuber"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <a href="https://open-llm-vtuber.github.io/docs/quick-start"><img src="https://img.shields.io/badge/Documentation-Quick_Start-blue" alt="Documentation"></a>
    <a href="https://github.com/orgs/Open-LLM-VTuber/projects/2"><img src="https://img.shields.io/badge/Roadmap-GitHub_Project-yellow" alt="Roadmap"></a>
    <a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</p>

<p align="center">
  <a href="https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md">中文README</a>
</p>


## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux, supporting both NVIDIA and non-NVIDIA GPUs, with CPU fallback.
*   **Offline Functionality:** Enjoy complete privacy and security with full offline mode using local models.
*   **Versatile Client Options:** Utilize either a web interface or a dedicated desktop client, which includes a transparent background pet mode for the AI companion to appear anywhere on your screen.
*   **Advanced Interaction:**
    *   Visual perception with camera, screen recording, and screenshots.
    *   Microphone voice interruption without headphones
    *   Touch feedback with clicks and drags
    *   Live2D expression control.
    *   Pet mode with transparent background, top-most display, and click-through functionality.
    *   AI thought display
    *   Proactive AI speaking
    *   Persistent chat logs.
    *   TTS Translation Support
*   **Extensive Model Support:** Compatible with a wide range of LLMs, ASR, and TTS solutions.
*   **Highly Customizable:**
    *   Simple module configuration.
    *   Customizable character appearance and persona through Live2D models and prompt modifications.
    *   Flexible Agent implementation for integration with any Agent architecture.
    *   Modular design for easy extension with custom modules.

## Demo

| <img src="assets/i1.jpg" alt="Demo Image 1"> | <img src="assets/i2.jpg" alt="Demo Image 2"> |
|:---:|:---:|
| <img src="assets/i3.jpg" alt="Demo Image 3"> | <img src="assets/i4.jpg" alt="Demo Image 4"> |

## Installation & Quick Start

Get started with Open-LLM-VTuber by following the [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start) guide in our documentation.

## Important Notices
*   **Active Development:** This project is in active development.
*   **HTTPS Configuration:** For remote access, configure HTTPS due to front-end microphone security requirements.

## Upgrade
Use `uv run update.py` to update.

## Uninstall
Most files and models are stored in the project folder. Check the installation guide for additional tools to remove.

## Contribute

For contribution guidelines, see the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview).

## Third-Party Licenses

### Live2D Sample Models Notice

This project includes Live2D sample models provided by Live2D Inc. These assets are licensed separately under the Live2D Free Material License Agreement and the Terms of Use for Live2D Cubism Sample Data. They are not covered by the MIT license of this project.

This content uses sample data owned and copyrighted by Live2D Inc. The sample data are utilized in accordance with the terms and conditions set by Live2D Inc. (See [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) and [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)).

Note: For commercial use, especially by medium or large-scale enterprises, the use of these Live2D sample models may be subject to additional licensing requirements. If you plan to use this project commercially, please ensure that you have the appropriate permissions from Live2D Inc., or use versions of the project without these models.

## Contributors

Thank you to our contributors and maintainers!

<a href="https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Open-LLM-VTuber/Open-LLM-VTuber" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=t41372/open-llm-vtuber&type=Date)](https://star-history.com/#t41372/open-llm-vtuber&Date)

## Related Projects

[ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant)
- Your Live2D desktop assistant powered by LLM! Available for both Windows and MacOS, it senses your screen, retrieves clipboard content, and responds to voice commands with a unique voice. Featuring voice wake-up, singing capabilities, and full computer control for seamless interaction with your favorite character.

## Get Started

Explore the source code and contribute on [GitHub](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)!