<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
</div>

<p align="center">Automate your daily and weekly tasks in Honkai: Star Rail with the March7thAssistant!</p>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

  [使用教程](https://m7a.top/#/assets/docs/Tutorial) | [FAQ](https://m7a.top/#/assets/docs/FAQ)
</div>

## Key Features

*   **Automated Daily Tasks:**  Automatically complete daily tasks, including stamina management, daily training, reward claiming, assignments, and "farming."
*   **Automated Weekly Tasks:** Automate weekly tasks like Simulated Universe, Echo of War and Forgotten Hall.
*   **SRGF Export and Auto-Dialogue:** Export your gacha history in SRGF format and enjoy automated in-game dialogue.
*   **Notifications:** Receive notifications upon completion of daily training and other tasks.
*   **Automated Triggers:** Automatically start tasks based on task refresh or stamina recovery thresholds.
*   **Customizable Actions:**  Receive sound alerts, automatically close the game, or even shut down your computer after tasks are finished.

## What is March7thAssistant?

March7thAssistant is a Windows-based automation tool designed to streamline your Honkai: Star Rail gameplay.  It helps you automate repetitive tasks, save time, and enhance your gaming experience.  This tool leverages existing open-source projects for functionality like Simulated Universe automation ([Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)) and farming ([Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)).

## Getting Started

1.  **Download:**  Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) section and download the latest release.
2.  **Run:**  Extract the downloaded archive and double-click `March7th Launcher.exe` to launch the graphical interface.
3.  **Configure:**  Adjust the settings according to your preferences.  See the [configuration file](assets/config/config.example.yaml) for details or use the GUI settings.

For automated scheduled execution or direct use, the terminal application is `March7th Assistant.exe`

Update the software by clicking the update button in the GUI settings or by double-clicking  `March7th Updater.exe`

## Important Notes

*   **Resolution:**  Requires a PC with a `1920x1080` resolution and the game running in a window or full-screen (HDR is not supported).
*   **Simulated Universe:**  Refer to the [Auto_Simulated_Universe documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Operation:** Explore [remote desktop](https://m7a.top/#/assets/docs/Background) for background operations and multi-monitor setups.
*   **Feedback:**  Report issues in the [Issues](https://github.com/moesnow/March7thAssistant/issues) section, discuss topics in [Discussions](https://github.com/moesnow/March7thAssistant/discussions), or contribute with [PRs](https://github.com/moesnow/March7thAssistant/pulls).

## Advanced Usage (for Developers)

If you're familiar with command-line tools:

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

The crop parameters can be obtained using the capture screenshot function in the assistant's toolbox.

The `python main.py` command supports arguments like `fight`, `universe`, and `forgottenhall`.

</details>

---

If you enjoy this project, consider supporting the author with a coffee ☕

Your support fuels the development and maintenance of the project!

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contribute

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

[Back to Top](#)