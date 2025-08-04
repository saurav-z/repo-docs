<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)

遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

---

## Automate Your Honkai: Star Rail Daily Tasks with March7th Assistant

March7th Assistant is a Windows-based automation tool designed to streamline your daily and weekly tasks in Honkai: Star Rail. [Check out the original repository](https://github.com/moesnow/March7thAssistant).

---

## Key Features

*   **Automated Daily Tasks**:
    *   Clear stamina.
    *   Complete daily training.
    *   Collect rewards.
    *   Handle commissions.
    *   Farm open-world resources ("锄大地").
*   **Automated Weekly Tasks**:
    *   Complete Simulated Universe runs.
    *   Clear Forgotten Hall.
    *   Fight Echo of War (历战余响).
*   **Gacha Record Export**:
    *   Supports the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard.
    *   Features **automatic dialog handling** for seamless operation.
*   **Notifications & Automation**:
    *   Receive **message notifications** upon completion of daily training and other tasks.
    *   **Auto-start** tasks when tasks refresh or stamina reaches a specified level.
    *   **Sound alerts, automatic game closure, or shutdown** upon task completion.

## Getting Started

### Download and Installation

1.  Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  Download the latest release.
3.  Extract the archive.
4.  Double-click the `March7th Launcher.exe` (with the March7th icon) to open the graphical user interface.

If you wish to use the **Task Scheduler** for automated runs or execute the **full program** directly, use the terminal icon `March7th Assistant.exe`.

### Updating

Check for updates within the GUI or by double-clicking `March7th Updater.exe`.

### Important Notes

*   Requires a **PC** with a `1920x1080` resolution window or full-screen game mode (HDR is *not* supported).
*   Configuration is available via the graphical interface or in the [configuration file](assets/config/config.example.yaml).
*   For background operation or multi-monitor setups, consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   For issues, please submit them on [Issues](https://github.com/moesnow/March7thAssistant/issues).  For discussion and questions, see [Discussions](https://github.com/moesnow/March7thAssistant/discussions).

## Development (For Advanced Users)

If you're a developer, follow these steps to run the code:

```cmd
# Installation (using venv is recommended)
git clone --recurse-submodules https://github.com/moesnow/March7thAssistant
cd March7thAssistant
pip install -r requirements.txt
python app.py
python main.py

# Update
git pull
git submodule update --init --recursive
```
<details>
<summary>Development Tips</summary>

Use the screenshot capture feature in the assistant toolbox to get the crop parameters.
You can pass arguments to `python main.py` like `fight`, `universe`, or `forgottenhall`.
</details>

---

If you like this project, you can buy the author a coffee ☕
![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant leverages the following open-source projects:

*   **Simulated Universe Automation:** [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Open World Farming Automation:** [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR (Optical Character Recognition):** [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Component Library:** [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)