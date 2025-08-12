<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <p>Automate your daily and weekly tasks in Honkai: Star Rail with the March7th Assistant!</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
  <br/>
  <a href="https://github.com/moesnow/March7thAssistant" target="_blank">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/moesnow/March7thAssistant?style=flat-square&color=4096d8" />
  </a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
  <!-- Internationalization -->
  **简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)
  <br/>
  <!-- Quick Start -->
  快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)
  <br/>
  <!-- FAQ -->
  遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)
</div>

## Key Features of March7thAssistant

March7thAssistant streamlines your Honkai: Star Rail gameplay, offering comprehensive automation and convenience:

*   **Automated Daily Tasks:** Clear stamina, complete daily training, collect rewards, handle commissions, and farm overworld resources.
*   **Weekly Content Automation:** Tackle Simulated Universe and Forgotten Hall challenges effortlessly.
*   **SRGF Export:** Easily export your in-game gacha history in the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, with **automatic dialogue** support.
*   **Notifications:** Receive message notifications upon the completion of daily training and other tasks.
*   **Automated Triggers:** Configure tasks to start automatically based on task refresh times or when stamina reaches a specified value.
*   **Post-Task Actions:** Customize actions like sound notifications, automatic game closure, or system shutdown after tasks are completed.

>  Uses external projects [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail).

For configuration details, refer to the [configuration file](assets/config/config.example.yaml) or the graphical interface settings.  🌟 If you like this project, please give it a star! (･ω･) 🌟  Join the QQ Group [Click to Join](https://qm.qq.com/q/LpfAkDPlWa)  or TG Group [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1).

## Screenshots

![March7thAssistant in Action](assets/screenshot/README.png)

## Important Notes

*   **PC Requirements:**  Ensure your game runs in a `1920x1080` resolution window or full-screen mode. HDR is not supported.
*   **Simulated Universe:** For more information on the Simulated Universe automation, consult the [project documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Usage:** Consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background) for background operation or multi-monitor setups.
*   **Reporting Issues:** Report any errors or issues via [Issues](https://github.com/moesnow/March7thAssistant/issues).  Discuss and ask questions in [Discussions](https://github.com/moesnow/March7thAssistant/discussions). PRs are welcome! [PRs](https://github.com/moesnow/March7thAssistant/pulls)

## Installation and Usage

1.  **Download:** Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) and download the latest release.
2.  **Run the Launcher:** Extract the archive and double-click `March7th Launcher.exe` (the icon with March7th) to start the graphical interface.
3.  **Scheduled Tasks/Full Run:** To use the task scheduler or execute the program directly, use the terminal icon `March7th Assistant.exe`.
4.  **Updates:**  Check for updates through the graphical interface settings, or by double-clicking `March7th Updater.exe`.

## Source Code

**If you're not a developer, it's recommended to download and install the pre-built version via the steps above.**

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
<summary>Development</summary>

Capture crop parameters using the screenshot capture function within the assistant toolbox.

Python main.py supports parameters such as fight/universe/forgottenhall.

</details>

---

If you find this project helpful, consider supporting the developer with a donation ☕

Your support fuels the project's development and maintenance 🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on the following open-source projects:

*   **Simulated Universe Automation:** [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Overworld Farming Automation:** [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR (Optical Character Recognition):** [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Framework:** [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)