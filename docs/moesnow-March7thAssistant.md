<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p><em>Automate your daily Honkai: Star Rail tasks and enjoy a seamless gaming experience!</em></p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release (latest by date)" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub all releases downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

</div>

**Quick Start:** [Tutorial](https://m7a.top/#/assets/docs/Tutorial) | **FAQ:** [FAQ](https://m7a.top/#/assets/docs/FAQ)

## Key Features of March7th Assistant

March7th Assistant is a powerful automation tool for *Honkai: Star Rail* players, designed to streamline your gameplay. Here are some of its key features:

*   **Automated Daily Tasks:**  Automatically complete daily training, claim rewards, manage commissions, and farm world exploration.
*   **Weekly Task Automation:** Automate weekly tasks such as Memory of Chaos and Simulated Universe.
*   **Automated Simulated Universe:**  Utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for streamlined Simulated Universe runs.
*   **Automated World Exploration:** Uses [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for automating world exploration tasks.
*   **抽卡记录导出:** Support [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, **自动对话**
*   **抽卡记录导出 & Auto-Dialogue:** Export your pull history and auto-dialogue features.
*   **Notification System:**  Receive notifications when tasks are complete.
*   **Automated Triggers:** Automatically start tasks based on in-game triggers, such as stamina recovery or task refresh.
*   **Customizable Actions:**  Configure sound alerts, auto-close game, or shutdown PC after task completion.

Find detailed configurations in the [configuration file](assets/config/config.example.yaml) or through the graphical user interface. 

🌟  Give the project a star if you like it!  🌟  Join the QQ group [click to join](https://qm.qq.com/q/LpfAkDPlWa) | Telegram group [click to join](https://t.me/+ZgH5zpvFS8o0NGI1)

## Example Interface

![README](assets/screenshot/README.png)

## Important Notes

*   **Platform:** Requires a Windows PC with a `1920x1080` resolution or full-screen game mode (HDR is not supported).
*   **Simulated Universe:**  See the [project documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for Simulated Universe-related information.
*   **Background Operations:** For background operation or multi-monitor setups, try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Report any errors via [Issues](https://github.com/moesnow/March7thAssistant/issues).  Discuss and ask questions in [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  PRs are welcome! [PRs](https://github.com/moesnow/March7thAssistant/pulls)

## Download and Installation

1.  Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  Download the latest release.
3.  Extract the archive.
4.  Double-click the `March7th Launcher.exe` icon to open the graphical interface.

To use the **Task Scheduler** or run directly, use `March7th Assistant.exe`.

Check for updates by clicking the button at the bottom of the graphical interface or by double-clicking `March7th Updater.exe`.

## Source Code Execution (Advanced)

**If you are a beginner, please use the Download and Installation steps above.**

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

The cropping coordinates can be obtained through the capture screenshot function within the assistant toolbox.

`python main.py` supports parameters like `fight`, `universe`, and `forgottenhall`.

</details>

---

If you find this project helpful, consider supporting the author with a coffee: ☕

Your support fuels the development and maintenance of the project!🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant utilizes the following open-source projects:

*   **Simulated Universe Automation:** [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **World Exploration Automation:** [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR:** [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Component Library:** [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors">
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

[Back to Top](#) - [View the Original Repo](https://github.com/moesnow/March7thAssistant)
```
Key improvements and SEO considerations:

*   **Clear Hook:** A concise opening sentence describing the primary function of the software.
*   **Keyword Optimization:**  Uses relevant keywords like "Honkai: Star Rail," "automation," "daily tasks," "weekly tasks," etc.
*   **Headings and Structure:** Uses clear headings to improve readability and SEO.
*   **Bulleted Lists:**  Highlights key features in a user-friendly format.
*   **Internal Linking:** Links to the tutorial, FAQ, and config file within the README itself.
*   **External Linking:** Links to the original repo,  related projects, and other useful resources.
*   **Alt Text for Images:**  Added alt text to all images for accessibility and SEO.
*   **Concise Language:**  Rewrote sections for clarity.
*   **Call to Action:** Encourages users to star the repo.
*   **Added a "Back to Top" link.**
*   **Incorporated "Contributors" and "Stargazers over time" badges to increase credibility and show activity.**