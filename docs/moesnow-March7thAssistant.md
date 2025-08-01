<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 星铁小助手
  </h1>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
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

快速上手，请访问：[使用教程](https://m7a.top/#/assets/docs/Tutorial)

遇到问题，请在提问前查看：[FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Automate Your Star Rail Adventures with March7thAssistant!

March7thAssistant is a powerful Windows application designed to automate various daily and weekly tasks in **Honkai: Star Rail**, saving you time and effort. Check out the [original repository](https://github.com/moesnow/March7thAssistant) for more details.

**Key Features:**

*   **Automated Daily Tasks:** Automatically complete daily training, claim rewards, handle commissions, and clear calyxes.
*   **Weekly Task Automation:** Tackle weekly challenges like Simulated Universe and Forgotten Hall.
*   **Automated Simulated Universe:** Integrates with [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for automated Simulated Universe runs.
*   **Automated Calyx (锄大地) Tasks:** Integrates with [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for automated Calyx runs.
*   **Pull Record Export:** Supports [SRGF](https://uigf.org/zh/standards/SRGF.html) standards for exporting pull records with **automatic dialogue**.
*   **Notification System:** Get notified when tasks are completed via message push notifications.
*   **Automated Triggers:** Automatically start tasks when refreshes or when your stamina reaches a specified value.
*   **Customizable Actions:** Configure actions such as sound notifications, automatic game closing, or system shutdown upon task completion.

**Configuration:**

*   Customize the application to your liking by modifying the [configuration file](assets/config/config.example.yaml) or through the intuitive graphical user interface.

🌟 If you find this project helpful, please give it a star! 🌟

## Screenshots

![Screenshot of March7thAssistant](assets/screenshot/README.png)

## Important Notes

*   Requires PC with a resolution of `1920*1080` in windowed mode or full-screen mode. HDR is not supported.
*   For Simulated Universe, see the documentation at [Auto_Simulated_Universe_Docs](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   For background operation or multi-monitor setups, consider using [remote desktop software](https://m7a.top/#/assets/docs/Background).
*   Report any issues on the [Issue tracker](https://github.com/moesnow/March7thAssistant/issues), and discuss ideas in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull Requests are welcome [PR](https://github.com/moesnow/March7thAssistant/pulls).

## Installation

1.  Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Extract the downloaded archive.
3.  Double-click the `March7th Launcher.exe` to launch the graphical interface.

If you need to schedule the program or run it via command line, use `March7th Assistant.exe`.

To check for updates, use the update button in the GUI or run `March7th Updater.exe`.

## Source Code Installation (For Developers)

If you're not a developer, please use the installation method above.

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
<summary>Development Related</summary>

You can get the crop parameters for cropping by using the screenshot capture function within the toolbox.

The python main.py command supports arguments like fight/universe/forgottenhall etc.
</details>

---

If you like this project, consider supporting the author with a coffee ☕

Your support fuels the development and maintenance of this project🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant leverages the following open-source projects:

*   **Simulated Universe Automation:** [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Calyx Automation:** [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR (Optical Character Recognition):** [PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Component Library:** [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)