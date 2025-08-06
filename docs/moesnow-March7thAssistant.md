<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant - Your Ultimate Honkai: Star Rail Companion
  </h1>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release (Latest by SemVer)" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub All Releases Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly by visiting: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

Find answers to your questions in the: [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## About March7th Assistant

**March7th Assistant is a powerful, Windows-based automation tool designed to streamline your daily and weekly tasks in Honkai: Star Rail.** (See original repo: [https://github.com/moesnow/March7thAssistant](https://github.com/moesnow/March7thAssistant))

## Key Features

*   **Automated Daily Tasks:** Clear stamina, complete daily training, collect rewards, manage commissions, and farm for resources.
*   **Weekly Content Automation:** Tackle Calyx and Simulated Universe challenges.
*   **Automated Combat:** Includes support for automated Simulated Universe using [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe). and farming using [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **Automated Resource Gathering:** Automatic farming for materials via [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail).
*   **Automated Card Extraction:** Support for SRGF standard card extraction and auto dialogue.
*   **Notifications:** Receive message notifications for completed tasks.
*   **Automated Triggers:** Automatic start based on task refresh or stamina recovery.
*   **Customizable Actions:** Configure sound notifications, automatic game closure, or shutdown upon task completion.
*   **Advanced Features:** Integration with SRGF for automated card data export, and customizable settings through the configuration file or GUI.

>   For details, please refer to the [configuration file](assets/config/config.example.yaml) or the graphical interface settings.  🌟 If you like it, give it a star! |･ω･) 🌟  Join the QQ group [Click to join](https://qm.qq.com/q/LpfAkDPlWa) TG group [Click to join](https://t.me/+ZgH5zpvFS8o0NGI1)

## Interface Showcase

![README](assets/screenshot/README.png)

## Important Notes

*   **Compatibility:** Requires PC with `1920*1080` resolution window or full-screen mode (HDR is not supported).
*   **Simulated Universe:** See the [project documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for the related project.
*   **Background Running:** For background operation or multi-monitor setups, consider using [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Support & Feedback:** Report issues on [GitHub Issues](https://github.com/moesnow/March7thAssistant/issues) and discuss on [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull requests are welcome! [PR](https://github.com/moesnow/March7thAssistant/pulls)

## Download and Installation

1.  Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Extract the downloaded archive.
3.  Double-click the `March7th Launcher.exe` icon to open the graphical interface.

**For scheduled tasks** or direct execution of the **full program**, use the terminal icon `March7th Assistant.exe`.

**Check for updates:** Click the update button in the GUI settings, or double-click `March7th Updater.exe`.

## Running from Source (Advanced Users)

If you're familiar with development:

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

Get the crop parameter's cropping coordinates using the capture screenshot function in the assistant toolbox.

The `python main.py` command supports arguments like fight/universe/forgottenhall, etc.

</details>

---

If you like this project, you can support the author with a coffee ☕

Your support motivates the author to develop and maintain the project 🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Material Farming Automation [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)