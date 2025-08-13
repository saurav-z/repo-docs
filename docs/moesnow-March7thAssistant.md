<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7thAssistant Logo">
    <br/>
    March7thAssistant · 三月七小助手 - Automated Honkai: Star Rail Tasks
  </h1>
  <p>
    <a href="https://trendshift.io/repositories/3892" target="_blank">
      <img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/>
    </a>
  </p>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
  <p>
    简体中文 | <a href="./README_TW.md">繁體中文</a> | <a href="./README_EN.md">English</a>
  </p>
  <p>
    Quick Start: <a href="https://m7a.top/#/assets/docs/Tutorial">Tutorial</a>
  </p>
  <p>
    FAQ: <a href="https://m7a.top/#/assets/docs/FAQ">FAQ</a>
  </p>
</div>

## ✨ **Automate Your Honkai: Star Rail Daily Grind with March7thAssistant!** ✨

March7thAssistant is a Windows-based automation tool designed to streamline your daily and weekly tasks in Honkai: Star Rail.  This tool, available on [GitHub](https://github.com/moesnow/March7thAssistant), allows you to automate tedious in-game activities, saving you time and effort.

**Key Features:**

*   ✅ **Daily Automation:** Automate daily tasks like clearing stamina, daily training, collecting rewards, dispatching assignments, and farming.
*   ✅ **Weekly Automation:** Automate weekly activities such as Simulated Universe and Forgotten Hall.
*   ✅ **Automated Actions:** Trigger tasks upon stamina recovery or specific time schedules.
*   ✅ **Automated Voice Prompts:** Receive voice notifications upon task completion.
*   ✅ **Automated Game Actions:** Automatically close the game or shut down your computer after tasks are done.
*   ✅ **SRGF Export:** Export your pulls using the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard.
*   ✅ **Automated Chat:** Auto-response feature included!
*   ✅ **Customizable Notifications:** Receive task completion notifications.

## 🖼️ Interface Showcase

![March7thAssistant Interface](assets/screenshot/README.png)

## ⚠️ Important Notes

*   **Platform:**  Requires a Windows PC.
*   **Resolution:**  Game must be running in a `1920x1080` resolution window or full screen (HDR is not supported).
*   **Simulated Universe:** See  [Auto_Simulated_Universe Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Usage:** For background operation or multi-monitor setups, consider [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Report bugs in the [Issues](https://github.com/moesnow/March7thAssistant/issues) and discuss in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Contributions via [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls) are welcome!

## ⬇️ Download and Installation

1.  Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  Download the latest release and unzip it.
3.  Double-click `March7th Launcher.exe` (the March7th icon) to open the graphical interface.
4.  For scheduled or direct execution, use `March7th Assistant.exe` in a terminal.
5.  To check for updates, use the update button in the GUI, or double-click `March7th Updater.exe`.

## 💻 Running from Source (For Advanced Users)

If you are not familiar with coding, please install via the download links above.

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
<summary>Development Information</summary>

You can get the crop parameters through the capture screenshot feature in the assistant's toolbox.

The `python main.py` command supports arguments like `fight`, `universe`, and `forgottenhall`.

</details>

---

If you like this project, you can support the author with a donation via WeChat:

![sponsor](assets/app/images/sponsor.jpg)

---

## 🤝 Related Projects

March7thAssistant leverages the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## 🧑‍🤝‍🧑 Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors"/>
</a>

## ⭐ Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)