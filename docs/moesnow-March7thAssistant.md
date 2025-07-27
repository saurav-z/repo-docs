<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <p>Automate your daily and weekly tasks in Honkai: Star Rail with March7thAssistant!</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)

遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Key Features of March7thAssistant

March7thAssistant is a Windows automation tool designed to streamline your Honkai: Star Rail gameplay.  It automates tasks for you, saving you time and effort.

*   **Automated Daily Tasks**:  Clears stamina, completes daily training, collects rewards, manages commissions, and farms for resources ("锄大地").
*   **Automated Weekly Tasks**:  Completes Simulated Universe, Memory of Chaos (忘却之庭), and Echo of War (历战余响).
*   **Automated Task Triggers**: Automates tasks on specific triggers, such as task refresh or when stamina reaches a specified value.
*   **Automated Draw Record Export**: Supports SRGF standard for easy sharing of your pulls, including automated dialog options.
*   **Notification System**:  Provides message push notifications to keep you informed on task completion, including sound alerts, automatic game closing, or shutdown of your computer.

**Get started now and simplify your Honkai: Star Rail experience! [Check it out on GitHub!](https://github.com/moesnow/March7thAssistant)**

> March7thAssistant utilizes the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) project for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for farming automation.

For more details, see the [configuration file](assets/config/config.example.yaml) or configure the graphical interface.  🌟 If you like it, please give it a star!|･ω･) 🌟｜Join our QQ group [here](https://qm.qq.com/q/LpfAkDPlWa) or our TG group [here](https://t.me/+ZgH5zpvFS8o0NGI1).

## Screenshots

![March7thAssistant Interface](assets/screenshot/README.png)

## Important Notes

*   **PC Only**:  Requires Windows PC with a `1920*1080` resolution or full-screen game display (HDR is not supported).
*   **Simulated Universe**:  Refer to the [Auto_Simulated_Universe documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Operation**:  For background operation or multi-monitor setups, consider using [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues**: Report any errors on the [Issue](https://github.com/moesnow/March7thAssistant/issues) page. Discuss and ask questions on the [Discussions](https://github.com/moesnow/March7thAssistant/discussions) page.
*   **Contributions Welcome**:  Contributions are welcome; please submit a [PR](https://github.com/moesnow/March7thAssistant/pulls).

## Installation and Usage

1.  **Download**: Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  **Extract**: Download and extract the latest release.
3.  **Launch**: Double-click `March7th Launcher.exe` to start the graphical interface.

**Automated Task Scheduling**: To schedule tasks or run the assistant directly, use `March7th Assistant.exe` (terminal icon).

**Check for Updates**:  Click the button at the bottom of the graphical interface settings or double-click `March7th Updater.exe` to check for updates.

## Source Code Installation (Advanced Users)

If you're comfortable with the command line, you can install and run the source code. (Otherwise, use the installation steps above.)

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

Capture crop parameters by using the screenshot capture tool within the assistant toolbox.

The `python main.py` command supports arguments such as fight/universe/forgottenhall.

</details>

---

If you find this project helpful, consider supporting the author with a coffee ☕ using the QR code below:

Your support fuels the development and maintenance of the project!🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI (Graphical User Interface) Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)