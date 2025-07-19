<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly with the [Usage Tutorial](https://m7a.top/#/assets/docs/Tutorial) or check the [FAQ](https://m7a.top/#/assets/docs/FAQ) if you have any issues.
</div>

## Automate Your Honkai: Star Rail Experience with March7th Assistant

March7th Assistant is a powerful Windows automation tool designed to streamline your daily tasks and enhance your gameplay in Honkai: Star Rail. Visit the [original repository](https://github.com/moesnow/March7thAssistant) for more information and updates.

### Key Features

*   **Automated Daily Tasks:**
    *   Clear stamina.
    *   Complete daily training.
    *   Claim rewards.
    *   Manage commissions.
    *   Automate "Simulated Universe" runs.
    *   Automate "Forgotten Hall" runs.
    *   Automate "Explore" (锄大地) runs.
*   **抽卡记录导出:** Support [SRGF](https://uigf.org/zh/standards/SRGF.html) Standard, **Automatic Chat**
*   **Notifications:** Receive notifications for task completion and stamina recovery.
*   **Automated Triggers:** Automatically start tasks upon daily refresh or when stamina reaches a specified value.
*   **Customizable Actions:** Sound notifications, automatic game closing, and even PC shutdown upon task completion.

### Get Started

1.  **Download and Install:** Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest). Extract the archive and run `March7th Launcher.exe`.

2.  **Run from Terminal:** For scheduled tasks or direct execution, use `March7th Assistant.exe`.

3.  **Update:** Update the application through the settings or by running `March7th Updater.exe`.

### Important Notes

*   **Resolution:** Requires a PC resolution of `1920x1080` with the game running in a window or full-screen mode (HDR not supported).
*   **Simulated Universe:** See the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) documentation for details.
*   **Background Operation:** Consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background) for background operation or multi-monitor setups.
*   **Issue Reporting:** Report any issues on the [Issues](https://github.com/moesnow/March7thAssistant/issues) page. Discuss and ask questions in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions) forum.
*   **Contributing:**  Contributions are welcome via [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).

### Development (For Advanced Users)

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

Use the capture screenshot function in the toolbox to get the crop parameters.
The `python main.py` command supports parameters like `fight/universe/forgottenhall`.

</details>

---

If you enjoy the project, consider supporting the author with a coffee:

![sponsor](assets/app/images/sponsor.jpg)

---

### Related Projects

March7th Assistant relies on these open-source projects:

*   **Simulated Universe Automation:** [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Explore Automation:** [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR:** [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Framework:** [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

### Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />
</a>

### Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)