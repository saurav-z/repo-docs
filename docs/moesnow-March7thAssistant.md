<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant - Your Automated Honkai: Star Rail Companion
  </h1>
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

**[简体中文](README.md)** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial) | FAQ: [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Automate Your Honkai: Star Rail Gameplay with March7th Assistant

March7th Assistant is a powerful Windows application designed to automate various tasks in Honkai: Star Rail, saving you time and effort. **Visit the [GitHub repository](https://github.com/moesnow/March7thAssistant) for more details!**

### Key Features:

*   **Daily Task Automation:** Automate daily tasks like clearing stamina, completing daily training, collecting rewards, and sending dispatches.
*   **Weekly Task Automation:** Automate weekly tasks, including Simulated Universe, Forgotten Hall, and Echo of War.
*   **Automated Battle:** Automate the whole battle process
*   **Automated Dialogue:** Automate the whole dialogue process.
*   **Automated Game Management:** Automatically launch or close game when task starts or completes.
*   **Automated Notification:** Receive notification for the training progress
*   **Automatic Trigger:** Trigger tasks upon resource refresh or stamina recovery, and sound notification
*   **SRGF Export:** Export your gacha records with SRGF format.

### Important Notes:

*   Requires a PC with a `1920x1080` resolution, and supports full screen
*   [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation.
*   [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for "Farming" automation.
*   Find configurations in the [config file](assets/config/config.example.yaml) or through the graphical interface.  |🌟If you like it, give it a star!･ω･) 🌟 | QQ Group: [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) | TG Group: [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1)

### Interface Preview

![README](assets/screenshot/README.png)

### Installation and Usage

1.  **Download:** Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) and download the latest release.
2.  **Run:** Extract the archive and double-click `March7th Launcher.exe` to open the graphical interface.
3.  **Scheduled Tasks/Direct Execution:** For scheduled tasks, use `March7th Assistant.exe` (terminal).
4.  **Update:** Check for updates by clicking the button in the graphical interface, or by double-clicking `March7th Updater.exe`.

### Advanced: Running from Source (For Developers)

If you're comfortable with coding, you can run the assistant from the source code:

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
<summary>Development Tips</summary>

Use the screenshot capture feature in the assistant's toolbox to get crop parameters.

Run `python main.py` with arguments like `fight`, `universe`, or `forgottenhall`.

</details>

---

If you find this project helpful, consider supporting the developer with a coffee! ☕

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant leverages the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   "Farming" Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />
</a>

## Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)