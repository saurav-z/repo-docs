<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 崩坏星穹铁道自动助手
  </h1>
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

## Introduction

**March7thAssistant is a powerful Windows application automating your daily and weekly tasks in *Honkai: Star Rail*, saving you time and effort!**  This project automates various gameplay elements, allowing you to enjoy more of the game with less repetitive clicking.  [Check it out on GitHub!](https://github.com/moesnow/March7thAssistant)

## Key Features

*   **Automated Daily Tasks:** Automate daily tasks such as clearing stamina, completing daily training, claiming rewards, handling commissions, and farming Calyxes (锄大地).
*   **Weekly Content Automation:**  Tackle weekly tasks including Simulated Universe and Forgotten Hall (忘却之庭) with ease.
*   **Automated Card Export:** Supports SRGF standard card export and includes auto-dialogue features.
*   **Notifications:** Receive notifications about task completion status, using message push functionality.
*   **Automated Triggers:**  Automatically start tasks upon daily reset or when stamina reaches a specified level.
*   **Customizable Actions:** Receive sound notifications and optionally automatically close the game or shut down your computer upon task completion.

##  Configuration & Resources

*   **Configuration:**  Customize your experience with the [configuration file](assets/config/config.example.yaml) or through the graphical user interface.
*   **Tutorial & FAQ:** Find detailed instructions in the [Tutorial](https://m7a.top/#/assets/docs/Tutorial) and [FAQ](https://m7a.top/#/assets/docs/FAQ) sections.
*   **Support:**  Join the QQ group [click to join](https://qm.qq.com/q/LpfAkDPlWa) or TG group [click to join](https://t.me/+ZgH5zpvFS8o0NGI1) for discussions and support.
*   **Contribute:**  Contribute to the project with [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).

## Screenshots

![March7thAssistant Interface](assets/screenshot/README.png)

## Important Notes

*   **Game Resolution:** Requires the game to be running on a PC with a `1920x1080` resolution in either windowed mode or full-screen (HDR not supported).
*   **Simulated Universe:** Refer to the [Auto_Simulated_Universe documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Operation:** For background operation or multi-monitor setups, consider using [Remote Local Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:**  Report any issues via the [Issues](https://github.com/moesnow/March7thAssistant/issues) section.  Discuss and ask questions in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions) section.

## Installation

1.  Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Extract the downloaded archive.
3.  Double-click `March7th Launcher.exe` (the March7th icon) to launch the graphical interface.

If you prefer to schedule tasks or run the assistant directly, use the `March7th Assistant.exe` file (terminal icon).

To check for updates, either click the update button in the graphical interface or double-click `March7th Updater.exe`.

## Source Code Usage

If you're new to this, follow the installation steps above.

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
<summary>Development Information</summary>

Capture crop parameters using the screenshot capture function in the assistant's toolbox.

You can pass arguments like fight/universe/forgottenhall to `python main.py`.

</details>

---

If you find this project helpful, consider supporting the developer with a coffee ☕ via WeChat.

Your support fuels the development and maintenance of this project!🚀

![sponsor](assets/app/images/sponsor.jpg)

---

##  Related Projects

March7thAssistant leverages these open-source projects:

*   **Simulated Universe Automation:** [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Calyx Farming Automation:** [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR Text Recognition:** [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Component Library:** [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)