<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p>Automate your Honkai: Star Rail daily tasks with this powerful and user-friendly assistant.</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Latest Release" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
  <a href="./README_EN.md">English</a> | <a href="./README_TW.md">繁體中文</a> | <b>简体中文</b>

  Quick start guide: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

  FAQ: [FAQ](https://m7a.top/#/assets/docs/FAQ)
</div>

## Key Features of March7th Assistant

March7th Assistant streamlines your Honkai: Star Rail gameplay, saving you time and effort.  Here's a breakdown of its powerful features:

*   **Automated Daily Tasks:**  Automatically completes daily activities such as:
    *   Stamina recovery
    *   Daily training
    *   Claiming rewards
    *   Completing commissions
    *   "Digging land" (锄大地)
*   **Automated Weekly Tasks:** Automates weekly tasks like:
    *   Forgotten Hall
    *   Simulated Universe
    *   Echo of War
*   **Automated Task Completion and Triggering:**
    *   Auto-launches tasks when stamina is full or when specified conditions are met.
    *   Provides sound notifications and optional game closing or system shutdown upon task completion.
*   **Comprehensive Support & Integration:**
    *   Supports [SRGF](https://uigf.org/zh/standards/SRGF.html) standard for exporting pull records with **automatic dialogue** options.
    *   Offers message push notifications for task completion.

  *   Integrates with [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for advanced automation.
*   **Customization:** Customize settings via the [configuration file](assets/config/config.example.yaml) or the graphical user interface.

    🌟 If you like it, please give us a star |･ω･) 🌟

    QQ group [Click to jump](https://qm.qq.com/q/LpfAkDPlWa) TG group [Click to jump](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![README](assets/screenshot/README.png)

## Important Notes

*   **Game Resolution:** Ensure the game is running on a PC with a `1920*1080` resolution or in full-screen mode. HDR is not supported.
*   **Simulated Universe:** Refer to the documentation for [Auto_Simulated_Universe](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for more details.
*   **Background Running:** For background operation or multi-monitor setups, consider using [remote desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Report any issues via the [Issues](https://github.com/moesnow/March7thAssistant/issues) page.  Discuss ideas and ask questions in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  PRs are welcome [PRs](https://github.com/moesnow/March7thAssistant/pulls)

## Download and Installation

1.  Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Download the latest release.
3.  Extract the archive.
4.  Double-click `March7th Launcher.exe` (the icon with March7th) to launch the GUI.

If you want to use the **Task Scheduler**, or directly execute the **full program**, use `March7th Assistant.exe` (terminal icon).

To check for updates, click the update button in the GUI, or double-click `March7th Updater.exe`.

## Running from Source (For Developers)

If you are a beginner, it's recommended to download and install via the above method.

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

You can get the crop parameters through the screenshot feature in the Assistant Toolbox.

`python main.py` supports arguments like fight/universe/forgottenhall
</details>

---

If you enjoy this project, you can support the developer with a coffee ☕

Your support motivates the developer to improve and maintain the project🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   "Digging Land" Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors">
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

**[Back to Top](#)** (Added for navigation)