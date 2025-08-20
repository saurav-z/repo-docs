<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant: Automate Your Daily Honkai: Star Rail Tasks
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

Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

Encountering issues? Check out the [FAQ](https://m7a.top/#/assets/docs/FAQ) before asking.

</div>

## About March7th Assistant

**March7th Assistant is your all-in-one automation tool for Honkai: Star Rail, designed to streamline your daily gameplay and save you time.**

[View the original repository on GitHub](https://github.com/moesnow/March7thAssistant)

## Key Features

*   **Daily Tasks Automation:** Automate daily activities such as claiming stamina, completing daily training, collecting rewards, handling commissions, and farming open-world resources (e.g., "锄大地").
*   **Weekly Tasks Automation:** Automate weekly tasks such as Simulated Universe and Memory of Chaos (Forgotten Hall).
*   **Automated Draw Record Export:** Supports the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard for easy draw record export and **automated conversation**.
*   **Notification System:** Receive **push notifications** upon completion of daily training and other tasks.
*   **Automated Launch & Shutdown:** Automatically starts tasks when stamina is refreshed or restored to a specific value, and can **play sounds, automatically close the game, or shut down your computer upon task completion.**

> Utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for farming automation.

For more details, see the [configuration file](assets/config/config.example.yaml) or the graphical user interface settings.  🌟 Give a star if you like it! 🌟  QQ Group [Click to join](https://qm.qq.com/q/LpfAkDPlWa)  TG Group [Click to join](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![README](assets/screenshot/README.png)

## Important Notes

*   Requires a **PC** with a `1920*1080` resolution window or full-screen game mode (HDR is not supported).
*   Simulated Universe related [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md)  [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)
*   For background operation or multiple monitors, you can try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background)
*   Report errors via [Issues](https://github.com/moesnow/March7thAssistant/issues).  Discuss and ask questions in [Discussions](https://github.com/moesnow/March7thAssistant/discussions).
*   Contributions are welcome via [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).

## Download and Installation

1.  Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Download the latest release.
3.  Unzip the downloaded file.
4.  Double-click `March7th Launcher.exe` (the March7th icon) to open the graphical interface.

If you need to schedule tasks using the **Task Scheduler** or run the **full operation** directly, you can use `March7th Assistant.exe` (the terminal icon).

To check for updates, click the button at the bottom of the graphical interface settings, or double-click `March7th Updater.exe`.

## Running from Source (For Developers)

If you're not a developer, please use the download and installation method above.

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
<summary>Development Related</summary>

You can get the crop parameters for cropping coordinates through the capture screenshot function in the toolbox.

The python main.py command supports parameters such as fight/universe/forgottenhall.

</details>

---

If you like this project, you can support the author by sending a coffee ☕.

Your support is the motivation for the author to develop and maintain the project 🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)