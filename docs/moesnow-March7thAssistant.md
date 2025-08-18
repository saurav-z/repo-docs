<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
</div>

<div align="center">
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Windows Platform" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**[简体中文](README_CN.md)** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Quick Start: [Tutorial](https://m7a.top/#/assets/docs/Tutorial) | FAQ: [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Automate Your Honkai: Star Rail Gameplay with March7th Assistant!

March7th Assistant is a Windows application designed to automate daily and weekly tasks in Honkai: Star Rail, saving you time and effort.  Check out the [original repository](https://github.com/moesnow/March7thAssistant) for the latest updates and contributions!

**Key Features:**

*   **Daily Task Automation:** Automates daily routines such as Stamina consumption, Daily Training, reward collection, and assignments.
*   **Weekly Task Automation:** Completes weekly tasks including Simulated Universe, Forgotten Hall, and Echo of War.
*   **Automated Card Export:** Exports your pull history in the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, with **automated dialogue**.
*   **Notification System:**  Provides **message notifications** for completed tasks, like Daily Training.
*   **Customizable Triggers:** Starts tasks automatically based on task refresh, stamina recovery, or other customizable conditions.
*   **Post-Task Actions:** Includes options for **sound notifications, automatic game closing, or system shutdown** upon task completion.

> Utilizes the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) project for Simulated Universe automation and the [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) project for exploration automation.

For detailed configuration options, refer to the [configuration file](assets/config/config.example.yaml) or the graphical user interface settings.  🌟 Give us a star if you like it! 🌟  Join the QQ group [here](https://qm.qq.com/q/LpfAkDPlWa) or the TG group [here](https://t.me/+ZgH5zpvFS8o0NGI1).

## Interface Showcase

![March7th Assistant Interface](assets/screenshot/README.png)

## Important Notes

*   Requires a **PC** with a `1920*1080` resolution and the game running in a window or full-screen mode (HDR is not supported).
*   Simulated Universe related documentation: [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)
*   For background operation and multi-monitor setups, try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   Report any issues in the [Issues](https://github.com/moesnow/March7thAssistant/issues) section. For discussions, use the [Discussions](https://github.com/moesnow/March7thAssistant/discussions) section. Feel free to submit [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).

## Download and Installation

1.  Download the latest release from the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  Extract the downloaded archive.
3.  Double-click `March7th Launcher.exe` (the March7th icon) to open the graphical user interface.

To schedule tasks using the **Task Scheduler** or run the **full application directly**, use `March7th Assistant.exe` (terminal icon).

To check for updates, click the update button in the GUI or double-click `March7th Updater.exe`.

## Running from Source (For Advanced Users)

If you are not a developer, please download and install the application as described above.

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

Use the screenshot capture function in the assistant toolbox to get the crop parameters.

You can pass arguments like `fight`, `universe`, or `forgottenhall` to `python main.py`.

</details>

---

If you enjoy this project, consider supporting the author with a coffee via WeChat. ☕

Your support motivates the author to develop and maintain this project.🚀

![Sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Exploration Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)
```
Key improvements and SEO optimizations:

*   **Clear Title:** Added "Automate Your Honkai: Star Rail Gameplay" to the title for SEO.
*   **Concise Hook:**  The one-sentence hook provides an immediate understanding of the project's value.
*   **Keyword Optimization:** Incorporated keywords like "Honkai: Star Rail," "automation," "tasks," "daily," "weekly," "Simulated Universe," and "pull history."
*   **Headings:** Structured the content logically with clear headings for readability.
*   **Bulleted Lists:** Used bullet points to highlight key features, making the information easy to scan.
*   **Action-Oriented Language:** Used active verbs (e.g., "Automate," "Completes," "Provides").
*   **Contextual Links:** Kept all relevant links, including those for the original repo, documentation, and related projects.
*   **Call to Action:** Encouraged users to give a star and engage with the project (QQ/TG group).
*   **Alt Text:** Added `alt` text to images for accessibility and SEO.
*   **Contributor Section:**  Maintained the contributor and stargazers over time sections, which adds social proof.
*   **Simplified Source Code Instructions:**  Improved the wording and clarity of instructions for running from source and added "For Advanced Users"