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
  <img alt="" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)

遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Automate Your Honkai: Star Rail Daily Tasks with March7thAssistant

March7thAssistant is a Windows-based automation tool designed to streamline your Honkai: Star Rail gameplay experience.  [Check it out on GitHub!](https://github.com/moesnow/March7thAssistant)

### Key Features

*   **Automated Daily Tasks:**  Efficiently completes daily tasks such as:
    *   Stamina clearing
    *   Daily training
    *   Claiming rewards
    *   Commission completion
    *   "Overworld" (锄大地) farming.
*   **Weekly Content Automation:**  Supports automation for weekly activities:
    *   Forgotten Hall
    *   Simulated Universe
    *   Memory of Chaos (忘却之庭)
*   **Automated Card Exporting:** Exports your card records in the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, and features **automated dialogue**.
*   **Customizable Notifications:**  Receive push notifications for completed tasks, like daily training.
*   **Intelligent Triggers:** Automatically launches tasks upon daily reset or when stamina reaches a specified value.
*   **Task Completion Alerts:**  Provides sound notifications, and can automatically close the game or shut down your computer upon task completion.

### How to Get Started

1.  **Download:** Get the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  **Run:** Unzip the downloaded file and double-click `March7th Launcher.exe` to open the graphical interface.
3.  **Command Line Usage:** For scheduled tasks or direct execution, use `March7th Assistant.exe` from the terminal.
4.  **Updates:** Check for updates through the graphical interface or run `March7th Updater.exe`.

### Important Notes

*   **System Requirements:**  Requires a Windows PC running the game in a `1920x1080` windowed or fullscreen mode (HDR is not supported).
*   **Simulated Universe:**  Utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe).
*   **"Overworld" Farming:**  Utilizes [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail).
*   **Troubleshooting:**  Report any issues via [Issues](https://github.com/moesnow/March7thAssistant/issues), or discuss them on [Discussions](https://github.com/moesnow/March7thAssistant/discussions).

### Development

For developers:

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
<summary>开发相关</summary>

获取 crop 参数表示的裁剪坐标可以通过小助手工具箱内的捕获截图功能

python main.py 后面支持参数 fight/universe/forgottenhall 等

</details>

---

If you like this project, you can support the author with a coffee ☕.

Your support is the motivation for the author to develop and maintain the project 🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Acknowledgements

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   "Overworld" Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)