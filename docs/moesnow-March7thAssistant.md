<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
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

Quick Start: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

Troubleshooting: [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

---

## Automate Your Honkai: Star Rail Daily Tasks with March7th Assistant

March7th Assistant is a powerful Windows application designed to automate various daily and weekly tasks in Honkai: Star Rail, saving you time and effort.  [Check out the original repo!](https://github.com/moesnow/March7thAssistant)

### Key Features:

*   **Daily Task Automation:** Automatically complete daily tasks, including stamina management, daily training, claiming rewards, dispatch missions, and "overworld" farming.
*   **Weekly Task Automation:** Automates weekly tasks like Simulated Universe, Echo of War and Forgotten Hall.
*   **Automated Actions:** Trigger actions upon task completion, such as sound notifications, auto-closing the game, or even system shutdown.
*   **SRGF Card Export:** Supports exporting your pull history in the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard and offers automatic dialogue.
*   **Customizable Triggers:**  Set up automatic launching of tasks when daily missions refresh or when stamina reaches a specific level.
*   **Notifications:** Receive notifications when daily training and other tasks are completed.

###  Getting Started

1.  **Download:** Download the latest release from the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  **Run:** Extract the downloaded archive and double-click `March7th Launcher.exe` to launch the graphical interface.  For scheduled tasks or full execution, use `March7th Assistant.exe`.
3.  **Update:** Check for updates using the button in the GUI or by running `March7th Updater.exe`.

### Important Notes

*   **Game Resolution:**  Requires the game to be running on PC in a `1920x1080` resolution window or full-screen mode (HDR is not supported).
*   **Simulated Universe:** Utilizes the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) project.
*   **"Overworld" Farming:**  Utilizes the [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) project.
*   **Configuration:** Customize settings via the [configuration file](assets/config/config.example.yaml) or the graphical user interface.
*   **Troubleshooting:** Report issues [here](https://github.com/moesnow/March7thAssistant/issues) and discuss them [here](https://github.com/moesnow/March7thAssistant/discussions).
*   **Background Execution:**  For background operation or multi-monitor setups, consider using [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).

### Code Execution

If you are a developer and want to run the code, follow these steps:

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

You can get the crop parameters for screenshot cropping by the assistant's toolbox.

python main.py supports the following optional arguments: fight/universe/forgottenhall ...

</details>

---

If you appreciate this project, consider supporting the author with a coffee! ☕

![sponsor](assets/app/images/sponsor.jpg)

---

### Related Projects

March7th Assistant leverages the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   "Overworld" Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

### Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors"/>
</a>

### Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)