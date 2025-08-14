<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7thAssistant Logo">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <p>Automate your daily and weekly tasks in Honkai: Star Rail with March7thAssistant!</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
  <br/>
  <a href="https://github.com/moesnow/March7thAssistant">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/moesnow/March7thAssistant?style=flat-square&color=4096d8" />
  </a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub all downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
  <br/>
  <a href="https://qm.qq.com/q/LpfAkDPlWa">
      <img src="https://img.shields.io/badge/QQ%E7%BE%A4-Click%20to%20Join-blue?style=flat-square&logo=tencentqq" alt="QQ Group">
  </a>
  <a href="https://t.me/+ZgH5zpvFS8o0NGI1">
      <img src="https://img.shields.io/badge/Telegram-Click%20to%20Join-blue?style=flat-square&logo=telegram" alt="Telegram Group">
  </a>
</div>

<br/>

<div align="center">
  <a href="./README_EN.md">English</a> | <a href="./README_TW.md">繁體中文</a> | 简体中文
</div>

## Key Features of March7thAssistant

March7thAssistant is your all-in-one companion for automating daily tasks in Honkai: Star Rail.

*   **Automated Daily Tasks:** Automatically clears stamina, completes daily training, collects rewards, handles commissions, and farms Calyxes.
*   **Weekly Content Automation:** Supports weekly content like Simulated Universe and Forgotten Hall.
*   **Automated Calyx Farming:** Farm Calyxes (锄大地) automatically.
*   **Automated Simulated Universe:** Automate Simulated Universe runs.
*   **Automated Rewards & Tasks:** Automatically collect rewards and complete specified tasks.
*   **Customizable Triggers:** Triggers tasks upon refresh or when stamina reaches a set value.
*   **Notifications & Alerts:** Receive message notifications upon task completion.
*   **Sound & Action Alerts:** Sound notifications, automatic game closing, or computer shutdown after tasks are complete.
*   **抽卡记录导出**：支持 [SRGF](https://uigf.org/zh/standards/SRGF.html) 标准、**自动对话**
*   **抽卡记录导出**：支持 [SRGF](https://uigf.org/zh/standards/SRGF.html) 标准、**自动对话**
*   **抽卡记录导出**：支持 [SRGF](https://uigf.org/zh/standards/SRGF.html) 标准、**自动对话**
*   **抽卡记录导出**：支持 [SRGF](https://uigf.org/zh/standards/SRGF.html) 标准、**自动对话**

For detailed usage instructions, please visit the [Tutorial](https://m7a.top/#/assets/docs/Tutorial).
Check the [FAQ](https://m7a.top/#/assets/docs/FAQ) for common questions and solutions.

## How to Get Started

1.  **Download:** Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page and download the latest version.
2.  **Installation:** Unzip the downloaded archive and double-click `March7th Launcher.exe` to launch the GUI.
3.  **Run in Terminal:**  For scheduled tasks or full automation, use `March7th Assistant.exe`.
4.  **Update:** Check for updates within the GUI or by running `March7th Updater.exe`.

## Important Notes

*   **Resolution:**  Requires a PC with a `1920x1080` resolution and the game running in either windowed or full-screen mode (HDR is not supported).
*   **Simulated Universe:** See the [Auto_Simulated_Universe Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Execution:** For background operation or multiple monitors, try [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Support:**  Report any issues on [Issues](https://github.com/moesnow/March7thAssistant/issues) and discuss in [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull Requests are welcome! [PR](https://github.com/moesnow/March7thAssistant/pulls)

## Source Code (For Developers)

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

The crop parameters can be obtained by using the screenshot capture function in the assistant toolbox.

The `python main.py` command supports arguments such as `fight`, `universe`, and `forgottenhall`.

</details>

---

If you find this project helpful, consider supporting the developer with a coffee:

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on these open-source projects:

*   **Auto Simulated Universe:** [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Fhoe-Rail (Calyx Farming Automation):** [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **PaddleOCR-json (OCR):** [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **PyQt-Fluent-Widgets (GUI):** [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributions

The project is hosted on GitHub at [https://github.com/moesnow/March7thAssistant](https://github.com/moesnow/March7thAssistant)

[View the contributors](https://github.com/moesnow/March7thAssistant/graphs/contributors)

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors" />
</a>

## Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)