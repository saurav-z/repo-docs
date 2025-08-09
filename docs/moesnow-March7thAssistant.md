<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p>Automate your daily and weekly tasks in Honkai: Star Rail with ease!</p>
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
  <a href="./README_TW.md">繁體中文</a> | <a href="./README_EN.md">English</a>
</div>

<br/>

**Quick start guide and documentation:** [使用教程](https://m7a.top/#/assets/docs/Tutorial)

**FAQ:** [FAQ](https://m7a.top/#/assets/docs/FAQ)

## Key Features

March7th Assistant streamlines your Honkai: Star Rail gameplay with these powerful features:

*   **Automated Daily Tasks:**  Automate daily tasks like stamina clearing, daily training, reward claiming, commissions, and farming (锄大地).
*   **Weekly Content Automation:** Complete weekly activities such as Simulated Universe (with integration using [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)), and Forgotten Hall.
*   **Automated Features:**
    *   Automated after task is completed, sounds prompt, auto close game or shutdown computer
    *   Task refreshes or stamina restored to a designated value, auto start
*   **Automatic Data Export:** Export your gacha history in the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, with **automatic dialogue** capabilities.
*   **Notifications:** Get notified on the completion of tasks like daily training.

>   For more details, see the [configuration file](assets/config/config.example.yaml) or set up through the GUI.  🌟 If you like this project, please give it a star! 🌟 |･ω･) 🌟|  Join the QQ group: [点击跳转](https://qm.qq.com/q/LpfAkDPlWa) or TG group: [点击跳转](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![Screenshot of March7th Assistant](assets/screenshot/README.png)

## Important Notes

*   **System Requirements:** Requires a **PC** running the game in a `1920*1080` resolution window or fullscreen mode (HDR is not supported).
*   **Simulated Universe:** Documentation for the Simulated Universe automation can be found [here](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [here](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Operation:** For background operation or multiple monitors, you can try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Please report any issues in the [Issue](https://github.com/moesnow/March7thAssistant/issues) section. Discuss in [Discussions](https://github.com/moesnow/March7thAssistant/discussions).

## Installation and Usage

1.  **Download:** Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  **Extract:** Download and extract the latest release.
3.  **Run:** Double-click `March7th Launcher.exe` (the icon with March7th image) to launch the GUI.

To run the assistant using **Task Scheduler** or execute it directly, use the terminal icon `March7th Assistant.exe`.

To check for updates, click the button at the bottom of the GUI settings or double-click `March7th Updater.exe`.

## Running from Source (For Developers)

If you're a developer or want to contribute:

```bash
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

You can get the cropping coordinates represented by the crop parameter through the screenshot capture function in the assistant toolbox.
python main.py supports parameters like fight/universe/forgottenhall etc.

</details>

---

If you like this project, you can support the author with a coffee ☕.

Your support is the motivation for the author to develop and maintain the project🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors">
</a>

## Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

[Back to Top](#) - Visit the [original repository](https://github.com/moesnow/March7thAssistant) for more information.