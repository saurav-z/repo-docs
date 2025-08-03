<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant - Your Automated Honkai: Star Rail Companion
  </h1>
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

[Quick Start Tutorial](https://m7a.top/#/assets/docs/Tutorial) | [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

---

## About March7thAssistant

**March7thAssistant is a powerful and easy-to-use automation tool designed to streamline your Honkai: Star Rail gameplay on PC, saving you time and effort.** Check out the original repo [here](https://github.com/moesnow/March7thAssistant).

---

## Key Features

*   **Automated Daily Tasks:**
    *   Clear stamina
    *   Daily Training
    *   Claim Rewards
    *   Commissions
    *   World Exploration (锄大地)
*   **Automated Weekly Tasks:**
    *   Echo of War (历战余响)
    *   Simulated Universe
    *   Forgotten Hall
*   **Automated Actions & Triggers:**
    *   SRGF standard support for **gacha record export** and **auto-dialogue**.
    *   **Automated Launch:** Starts tasks automatically upon refresh or stamina recovery to a specific value.
    *   **Notifications:** Get notified via message push after completing tasks
    *   **Action completion notification:** Sound alerts, auto-close game or shutdown.
*   **Customizable and Configurable:**
    *   Easily adjust settings via a graphical user interface or modify the `config.example.yaml` file.

>   This project utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for World Exploration automation.

## Screenshots

![Screenshot of March7thAssistant Interface](assets/screenshot/README.png)

## Important Notes

*   **PC Only:**  Requires a PC running the game at `1920*1080` resolution, either in full-screen or windowed mode (HDR not supported).
*   **Simulated Universe Documentation:** Find more information in the [Auto_Simulated_Universe Docs](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Execution:**  For background operation and multi-monitor setups, consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Feedback & Support:** Report issues in the [GitHub Issues](https://github.com/moesnow/March7thAssistant/issues), discuss features or problems in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions), and feel free to submit [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).

## Installation and Usage

1.  **Download:**  Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page and download the latest release.
2.  **Run:** Unzip the downloaded archive and double-click `March7th Launcher.exe` to open the graphical interface.
3.  **Advanced:** To use the **Task Scheduler** or run the full application directly, use `March7th Assistant.exe`.
4.  **Updates:** Check for updates by clicking the button at the bottom of the graphical interface, or double-click `March7th Updater.exe`.

## Source Code Run (For Developers)

**If you're not a developer, it's recommended to use the download installation method above.**

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
<summary>Development Notes</summary>

Use the capture screenshot function in the helper toolbox to get crop parameters.

You can run `python main.py` with arguments such as `fight/universe/forgottenhall`.

</details>

---

If you enjoy this project, consider supporting the developer with a coffee ☕ (link to sponsor image)

Your support fuels the development and maintenance of this project!🚀

---

## Related Projects

March7thAssistant relies on these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   World Exploration Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors" />
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)