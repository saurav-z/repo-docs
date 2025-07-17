<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 崩坏：星穹铁道 小助手
  </h1>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)

遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## About March7thAssistant

**March7thAssistant is your all-in-one desktop assistant for automating daily tasks and enhancing your *Honkai: Star Rail* gameplay on Windows.**  Automate your daily routines and streamline your gaming experience with this powerful tool!

✨ **[View the original repository on GitHub](https://github.com/moesnow/March7thAssistant)** ✨

## Key Features

*   **Automated Daily Tasks:**
    *   Clear Stamina
    *   Complete Daily Training
    *   Claim Rewards
    *   Handle Assignments
    *   Automated "锄大地" (Ground Farming)
*   **Automated Weekly Tasks:**
    *   Echo of War Automation
    *   Simulated Universe Automation
    *   Memory of Chaos Automation
*   **抽卡记录 Export & Auto-Dialogue:** Supports the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard for exporting gacha records and includes automated dialogue functionality.
*   **Notifications:** Receive message notifications for task completion (e.g., Daily Training).
*   **Automation Triggers:** Automatically starts tasks upon refresh or when stamina reaches a specified value.
*   **Post-Task Actions:** Customizable actions after task completion, including sound notifications, game closure, or system shutdown.

## UI Preview

![March7thAssistant Interface](assets/screenshot/README.png)

## Important Notes

*   **System Requirements:** Requires a Windows PC and a game resolution of `1920x1080` in either windowed or full-screen mode (HDR is not supported).
*   **Simulated Universe:**  Refer to these resources for the Simulated Universe module:  [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) & [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)
*   **Background Running:**  For background operation or multi-monitor setups, consider using [remote desktop solutions](https://m7a.top/#/assets/docs/Background).
*   **Support:**
    *   Report issues in the [Issues](https://github.com/moesnow/March7thAssistant/issues) section.
    *   Discuss and ask questions in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).
    *   Consider contributing through [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).
*   **群聊随缘看，欢迎 [PR](https://github.com/moesnow/March7thAssistant/pulls)**

## Installation

1.  Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Unzip the downloaded archive.
3.  Double-click `March7th Launcher.exe` to open the graphical interface.

**For scheduled tasks:** If you need to schedule tasks using the **Task Scheduler** or execute a "full run", use the terminal application `March7th Assistant.exe`.

**For updates:** Click the update button within the GUI, or double click the  `March7th Updater.exe`.

## Source Code Installation (For Developers/Advanced Users)

***Note:** If you're a beginner, please use the pre-built installer.*

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

Use the screenshot capture tool within the assistant to obtain crop parameters.

You can specify command-line arguments after `python main.py`, such as `fight`, `universe`, or `forgottenhall`.

</details>

---

If you like this project, consider supporting the author with a coffee ☕ via WeChat! Your support fuels the development and maintenance of this project.

![sponsor](assets/app/images/sponsor.jpg)

---

## Dependencies

March7thAssistant leverages these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Ground Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)