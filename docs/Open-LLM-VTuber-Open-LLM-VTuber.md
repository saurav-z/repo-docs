<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.jpg">
  <img src="./assets/banner.jpg" alt="Open-LLM-VTuber Banner" width="100%">
</picture>

# Open-LLM-VTuber: Your AI Companion with a Live2D Avatar

> Create your own interactive AI companion with a customizable Live2D avatar using Open-LLM-VTuber, a fully offline, cross-platform project.  [Get Started Here](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)!

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

[English README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.md) | [中文README](https://github.com/t41372/Open-LLM-VTuber/blob/main/README.CN.md)

[Documentation](https://open-llm-vtuber.github.io/docs/quick-start) | [![Roadmap](https://img.shields.io/badge/Roadmap-GitHub_Project-yellow)](https://github.com/orgs/Open-LLM-VTuber/projects/2)

<a href="https://trendshift.io/repositories/12358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12358" alt="t41372%2FOpen-LLM-VTuber | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.  Supports NVIDIA and non-NVIDIA GPUs, offering CPU fallback and GPU acceleration where available.
*   **Offline Mode:** Utilize local models for private conversations and complete offline functionality – no internet needed!
*   **Flexible Client Options:** Enjoy both web and desktop client modes, with desktop pet mode for a customizable on-screen companion.
*   **Advanced Interaction:**
    *   Visual perception through camera, screen recording, and screenshots.
    *   Voice interruption without echo and touch feedback.
    *   Live2D expression mapping and customizable pet mode.
    *   AI's inner thought visualization.
    *   Proactive AI speaking and chat log persistence.
    *   TTS translation support.
*   **Extensive Model Support:** Integrates a wide range of LLMs (Ollama, OpenAI, Gemini, etc.), ASR (sherpa-onnx, Whisper, etc.), and TTS (sherpa-onnx, Coqui-TTS, Edge TTS, etc.) models.
*   **Highly Customizable:**
    *   Simple module configuration for easy customization.
    *   Customizable AI companion appearance and personality through Live2D models and prompts.
    *   Flexible Agent implementation for integrating different architectures.
    *   Modular design for adding LLMs, ASR, and TTS modules.

## Demo

| ![](assets/i1.jpg) | ![](assets/i2.jpg) |
|:---:|:---:|
| ![](assets/i3.jpg) | ![](assets/i4.jpg) |

## Installation and Quick Start

Refer to the [Quick Start Guide](https://open-llm-vtuber.github.io/docs/quick-start) to get your AI companion up and running quickly.

## Update
> :warning: `v1.0.0` has breaking changes and requires re-deployment. You *may* still update via the method below, but the `conf.yaml` file is incompatible and most of the dependencies needs to be reinstalled with `uv`. For those who came from versions before `v1.0.0`, I recommend deploy this project again with the [latest deployment guide](https://open-llm-vtuber.github.io/docs/quick-start).

Use `uv run update.py` to update if you installed any versions later than `v1.0.0`.

## Uninstall
Most files, including Python dependencies and models, are stored in the project folder.

However, models downloaded via ModelScope or Hugging Face may also be in `MODELSCOPE_CACHE` or `HF_HOME`. While we aim to keep them in the project's `models` directory, it's good to double-check.

Review the installation guide for any extra tools you no longer need, such as `uv`, `ffmpeg`, or `deeplx`.

## Contributing

We welcome contributions!  Check out the [development guide](https://docs.llmvtuber.com/docs/development-guide/overview) to learn how you can contribute.

## Related Projects

*   [ylxmf2005/LLM-Live2D-Desktop-Assitant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) -  A Live2D desktop assistant powered by LLM!

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