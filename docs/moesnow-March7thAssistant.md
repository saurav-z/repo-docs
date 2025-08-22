<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <p><b>Automate your Honkai: Star Rail daily tasks with March7thAssistant, a powerful Windows automation tool!</b></p>
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
  <a href="./README_EN.md">English</a> | <a href="./README_TW.md">繁體中文</a> | 简体中文
  <br>
  Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)
  <br>
  Find answers in the [FAQ](https://m7a.top/#/assets/docs/FAQ)
</div>

## Key Features

March7thAssistant streamlines your Honkai: Star Rail gameplay with these key features:

*   **Automated Daily Tasks:** Clears stamina, completes daily training, claims rewards, handles commissions, and farms for resources (锄大地).
*   **Automated Weekly Tasks:** Completes Simulated Universe, Forgotten Hall (忘却之庭), and Echo of War (历战余响).
*   **Automated Simulated Universe and Forgotten Hall tasks.**
*   **Automated resource farming (锄大地)**.
*   **Automated actions:** Refreshes tasks, and automates tasks upon stamina recovery or specified values.
*   **抽卡记录导出:** Supports [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, and includes **auto-dialogue**.
*   **Notifications:** Push notifications for daily training completion and other events.
*   **Customizable Actions:** Sound notifications, automatic game closing, and shutdown options upon task completion.

> Leverages [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for resource farming.

Explore the [Configuration File](assets/config/config.example.yaml) or the graphical interface for detailed settings.  🌟 If you like this project, please give it a star! |･ω･) 🌟 Join the QQ group [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) or the TG group [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1) for support and discussions.

## Screenshots

![March7thAssistant Interface](assets/screenshot/README.png)

## Important Notes

*   **System Requirements:** Requires a **PC** running the game in a `1920*1080` resolution window or full-screen mode. (HDR is not supported).
*   **Simulated Universe:** See [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for related information.
*   **Background Execution:** For background operation or multi-monitor setups, consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Report bugs and issues via [GitHub Issues](https://github.com/moesnow/March7thAssistant/issues). Discuss and ask questions in [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull requests are welcome via [PRs](https://github.com/moesnow/March7thAssistant/pulls).

## Installation and Setup

1.  **Download:** Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  **Extract:** Unzip the downloaded archive.
3.  **Run:** Double-click the `March7th Launcher.exe` (with the March7th icon) to launch the graphical interface.

To run the application with the **Task Scheduler** or directly execute the **full run**, use `March7th Assistant.exe` (with the terminal icon).

To check for updates, either click the update button in the GUI or double-click `March7th Updater.exe`.

## Running from Source (For Advanced Users)

If you're comfortable with coding, follow these steps:

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

You can obtain crop parameter coordinates using the capture screenshot function within the Assistant Toolkit.

The `python main.py` command supports arguments like `fight`, `universe`, and `forgottenhall`.

</details>

---

If you appreciate this project, consider supporting the author with a coffee ☕ through WeChat.

Your support fuels the ongoing development and maintenance of this project! 🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant is built upon and benefits from these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Resource Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contribution and Community

*   [GitHub Repository](https://github.com/moesnow/March7thAssistant)
*   [View the source code on GitHub](https://github.com/moesnow/March7thAssistant)
*   [GitHub Issues](https://github.com/moesnow/March7thAssistant/issues)
*   [Discussions](https://github.com/moesnow/March7thAssistant/discussions)
*   [Releases](https://github.com/moesnow/March7thAssistant/releases/latest)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors">

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)