<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Latest Release" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
    <a href="https://github.com/moesnow/March7thAssistant">
        <img alt="GitHub Repo Link" src="https://img.shields.io/badge/GitHub-moesnow%2FMarch7thAssistant-blue?style=flat-square&logo=github">
    </a>
  **简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

  快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)
  遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)
</div>

## Automate Your Honkai: Star Rail Daily Tasks with March7thAssistant

March7thAssistant is a Windows application designed to automate various daily and weekly tasks in Honkai: Star Rail, saving you time and effort.

**Key Features:**

*   **Daily Task Automation:** Automate daily routines like clearing stamina, completing daily training, claiming rewards, and managing commissions.
*   **Weekly Task Automation:** Automate weekly tasks like Simulated Universe, Forgotten Hall, and Echo of War.
*   **Automated Card Record Export:** Supports SRGF standard for easy card history export and automatic dialogue.
*   **Notification System:** Receive notifications for completed daily training and other tasks.
*   **Scheduled Automation:** Automatically start tasks upon refresh or when stamina reaches a specified value.
*   **Customizable Actions:** Configure actions upon task completion, such as sound notifications, automatic game closing, or computer shutdown.

> This project utilizes the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) projects for Simulated Universe and Overworld farming automation, respectively.

For detailed configuration options, refer to the [configuration file](assets/config/config.example.yaml) or the graphical user interface.  🌟 If you enjoy this project, please give it a star!🌟  Join the QQ group: [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) or TG group: [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1).

## Screenshots

![README](assets/screenshot/README.png)

## Important Notes

*   Requires a **PC** with a `1920x1080` resolution and the game running in a window or full-screen mode (HDR is not supported).
*   For Simulated Universe related documentation: [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)
*   For background operation or multi-monitor setups, try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   Report issues in the [Issues](https://github.com/moesnow/March7thAssistant/issues) section, and discuss in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions). PRs are welcome: [PRs](https://github.com/moesnow/March7thAssistant/pulls)

## Installation and Usage

1.  **Download:** Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page and download the latest release.
2.  **Extract:** Extract the downloaded archive.
3.  **Run:** Double-click `March7th Launcher.exe` (the icon with March 7th's image) to open the GUI.

**For Scheduled Tasks (Task Scheduler) or Full Execution:** Use the terminal icon `March7th Assistant.exe`.

**Update:** To check for updates, click the button at the bottom of the GUI settings or double-click `March7th Updater.exe`.

## Running from Source (Advanced)

If you're a beginner, it's recommended to use the pre-built releases above.

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

You can get the crop parameters using the screenshot capture function in the helper toolbox.

The `python main.py` command supports arguments like `fight`, `universe`, `forgottenhall`, etc.
</details>

---

If you appreciate the project, consider supporting the developer with a coffee:

![sponsor](assets/app/images/sponsor.jpg)

---

## Dependencies & Acknowledgements

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Overworld Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)
```
Key improvements and SEO optimizations:

*   **Clear Title and Introduction:**  The title is kept but the intro sentence is added as a hook.
*   **Strategic Keyword Placement:** Keywords like "Honkai: Star Rail," "automation," "daily tasks," and specific task names are incorporated naturally.
*   **Concise Key Feature Section:** Key features are listed with bullet points for easy readability.
*   **Download and Usage Section:** The installation and usage instructions are enhanced.
*   **Strong Call to Action:**  Encourages users to explore the project and contribute.
*   **Contributor and Star History:** Added a contributor section and star history section for social proof.
*   **GitHub Link:** Added a GitHub Repo Link badge.
*   **Clear Headings and Structure:** The use of headings and subheadings makes the content easier to scan and digest.
*   **SEO-Friendly Language:**  Uses common search terms and phrases related to the game and automation.
*   **Link Back to Original:** The link to the original repo is included.