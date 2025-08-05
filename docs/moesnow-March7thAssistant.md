<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 崩坏：星穹铁道 自动化助手
  </h1>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
  <br/>
  <a href="https://github.com/moesnow/March7thAssistant" target="_blank">  <img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift"  /></a>

</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial) | 遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## 🚀 Automate Your Honkai: Star Rail Daily Tasks with March7thAssistant! 🚀

March7thAssistant is a Windows-based automation tool designed to streamline your daily and weekly tasks in Honkai: Star Rail. It helps you save time and effort by automating repetitive gameplay elements.

**Key Features:**

*   **Daily Task Automation:** Automate daily training, claim rewards, complete commissions, and clear Calyxes (daily quests).
*   **Weekly Task Automation:** Automate Simulated Universe and Forgotten Hall.
*   **Automated Battle and Task Execution:** Set tasks to run automatically after specified conditions, such as time of day, and more.
*   **Automated Dialogue and SRGF Export:** Automatically handles dialogue, and export your pull history to the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard.
*   **Notifications:** Receive notifications for task completion via your preferred methods.
*   **Automatic Startup Triggers:** Configure tasks to automatically start when missions refresh or stamina recovers to a specified value.
*   **Audio & Visual Cues:** Receive audio cues and game/PC shut down options upon task completion.

**Find out more in the [configuration file](assets/config/config.example.yaml) or within the graphical user interface.  Star the project if you like it!🌟|･ω･) 🌟｜QQ群 [点击跳转](https://qm.qq.com/q/LpfAkDPlWa) TG群 [点击跳转](https://t.me/+ZgH5zpvFS8o0NGI1)**

## 🖼️ Interface Showcase

![README](assets/screenshot/README.png)

## ⚠️ Important Notes

*   **System Requirements:** Requires a Windows PC with a `1920x1080` resolution window or full-screen game display (HDR is not supported).
*   **Simulated Universe Automation:** Relies on the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) project.
*   **Calyx Automation:** Uses the [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) project for automation.
*   **Background Operation:** For running in the background or with multiple monitors, consider using [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Issue Reporting & Discussion:** Report bugs and discuss features in the [Issue](https://github.com/moesnow/March7thAssistant/issues) and [Discussions](https://github.com/moesnow/March7thAssistant/discussions) sections.  Pull requests are welcome [PR](https://github.com/moesnow/March7thAssistant/pulls).

## ⬇️ Download & Installation

1.  Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  Download the latest release.
3.  Extract the contents of the archive.
4.  Double-click the `March7th Launcher.exe` file (with the March7th icon) to launch the graphical user interface.

**For Scheduled Tasks:** To schedule the tool or run the **full operation**, use the `March7th Assistant.exe` file (terminal icon).

**Updating:** Check for updates by clicking the button at the bottom of the GUI, or double-clicking `March7th Updater.exe`.

## 💻 Source Code Usage (For Advanced Users)

If you're familiar with Python and Git, you can run the source code directly.

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
<summary>Development Details</summary>

To obtain crop parameters, use the screenshot capture function within the toolkit.
The `python main.py` command supports arguments for tasks like `fight`, `universe`, and `forgottenhall`.

</details>

---

<div align="center">
If you like this project, you can buy the author a coffee.☕

Your support is the driving force behind the development and maintenance of the project🚀
</div>
<div align="center">
![sponsor](assets/app/images/sponsor.jpg)
</div>

---

## 🤝 Dependencies & Credits

March7thAssistant is built on the shoulders of these amazing open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Calyx Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## 🤝 Contributors

[<img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />](https://github.com/moesnow/March7thAssistant/graphs/contributors)

## 📈 Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)