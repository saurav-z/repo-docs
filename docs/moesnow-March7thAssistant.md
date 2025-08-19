<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant: Automate Your Honkai: Star Rail Daily Tasks
  </h1>
  <p>A convenient tool to streamline your daily Honkai: Star Rail gameplay.</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

  <a href="https://github.com/moesnow/March7thAssistant">
    <img src="https://img.shields.io/badge/View%20on%20GitHub-blue?style=flat-square&logo=github" alt="View on GitHub">
  </a>
  <br/>
  **简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)
  <br/>
  <a href="https://m7a.top/#/assets/docs/Tutorial">Quick Start Tutorial</a> | <a href="https://m7a.top/#/assets/docs/FAQ">FAQ</a>
</div>

## Key Features

March7th Assistant simplifies your Honkai: Star Rail experience with these key features:

*   **Automated Daily Tasks**:  Efficiently clear stamina, complete daily training, collect rewards, manage assignments, and farm the overworld.
*   **Automated Weekly Tasks**: Automate Simulated Universe and Forgotten Hall runs.
*   **Automated Actions**: Auto start, auto-close game or shut down PC after completing tasks.
*   **Automated Push Notifications**: Receive notifications about task completion.
*   **Automated Triggers**:  Start tasks automatically when stamina is replenished or reaches a specific value.
*   **抽卡记录导出**: Support [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, **auto-dialog**.
*   **抽卡记录导出**: Support [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, **auto-dialog**.

  <br/>
  > Utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for farming the overworld.

## Screenshots

![Main Interface](./assets/screenshot/README.png)

## Installation and Usage

1.  **Download:** Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) and download the latest version.
2.  **Run:** Extract the downloaded archive and double-click `March7th Launcher.exe` to launch the graphical interface.
3.  **Advanced Usage**: Use the terminal icon `March7th Assistant.exe` for scheduled tasks.
4.  **Update:** Check for updates via the button in the GUI or by double-clicking `March7th Updater.exe`.

## Important Notes

*   **Resolution:** Requires a PC with a `1920x1080` resolution.
*   **HDR:** HDR is not supported.
*   **Troubleshooting:** Report issues [here](https://github.com/moesnow/March7thAssistant/issues). Discuss and ask questions [here](https://github.com/moesnow/March7thAssistant/discussions).
*   **Background Operation:**  For background operation and multi-monitor setups, consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Simulation Universe Docs:** [Simulation Universe Docs](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md)
*   **Simulation Universe Q&A:** [Simulation Universe Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)

## Source Code

For developers and advanced users:

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
Get crop parameters from capture screenshots within the assistant's toolbox.

Use parameters such as `fight/universe/forgottenhall` in `python main.py`.
</details>

---

If you find this project helpful, consider supporting the developer with a coffee!
![sponsor](assets/app/images/sponsor.jpg)
---

## Related Projects

March7th Assistant is built upon these excellent open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Overworld Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors" />
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)