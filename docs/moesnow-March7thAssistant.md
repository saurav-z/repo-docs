<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p><em>Automate your daily Honkai: Star Rail tasks with March7th Assistant!</em></p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial) |  [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Key Features

March7th Assistant is your all-in-one automation tool for Honkai: Star Rail, streamlining your gameplay and saving you time.

*   **Automated Daily Tasks**:  Complete daily training, receive rewards, handle commissions, and clear Calyx.
*   **Automated Weekly Tasks**:  Automate Memory of Chaos and Simulated Universe runs.
*   **SSRGF Export**: Supports [SRGF](https://uigf.org/zh/standards/SRGF.html) standard for card exporting. Includes **auto-dialogue**.
*   **Notification System**:  Receive push notifications upon completion of daily tasks.
*   **Smart Triggers**: Automatically starts tasks when the stamina or other conditions are met.
*   **Customizable Actions**: Sound notifications, auto-close game, or shutdown after task completion.

>  This project utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for Calyx clearing.

See the [configuration file](assets/config/config.example.yaml) or graphical interface settings for more details.  🌟  Like this project? Give it a star!  🌟  Join the QQ Group: [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) | TG Group: [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1)

##  GUI Showcase

![GUI Screenshot](assets/screenshot/README.png)

## Important Notes

*   **PC Requirements**: Requires a PC running the game in a `1920*1080` resolution window or full-screen mode (HDR not supported).
*   Simulated Universe Documentation: [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md)  [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)
*   **Background Running**:  For background operation or multiple monitors, try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues**:  Report any issues in the [Issue Tracker](https://github.com/moesnow/March7thAssistant/issues) and discuss them in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions). The group chats are less active.  Contributions are welcome via [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls)!

## Download & Installation

1.  Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Download the latest release.
3.  Unzip the archive.
4.  Double-click `March7th Launcher.exe` (the March7th icon) to open the graphical interface.

To schedule tasks using the **Task Scheduler** or to run the **full automation directly**, use the terminal icon's `March7th Assistant.exe`.

To check for updates, click the button at the bottom of the graphical interface settings, or double-click `March7th Updater.exe`.

## Running from Source (For Developers)

If you are new to this, it's recommended to install and use the pre-built releases above.

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

You can use the capture screenshot function in the assistant toolbox to get the crop parameters for cropping coordinates.

The following parameters are supported after python main.py: fight/universe/forgottenhall etc.

</details>

---

If you like this project, you can support the author with a coffee via WeChat:

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Calyx Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors">
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

[Back to Top](#) (Link back to original repo if needed)
```

Key improvements and SEO optimizations:

*   **Concise Hook**:  A compelling one-sentence introduction at the beginning.
*   **Clear Headings:**  Uses consistent and descriptive headings.
*   **Keyword Optimization**: Includes keywords like "Honkai: Star Rail," "automation," and task names.
*   **Bulleted Key Features**: Uses a bulleted list for easy readability.
*   **Call to Action**: Encourages users to star the project and join the community.
*   **Contextual Links**: Links are integrated naturally within the text.
*   **Alt Text for Images:** Added `alt` text to images for accessibility and SEO.
*   **Simplified "Running from Source" Section**:  Clearer instructions.
*   **Back to top link** To make navigation simple.