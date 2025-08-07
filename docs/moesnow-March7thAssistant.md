<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p>Automate your Honkai: Star Rail daily tasks with March7th Assistant!</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

Find answers to your questions: [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Key Features

March7th Assistant is a powerful automation tool designed to streamline your Honkai: Star Rail gameplay. Here's what it offers:

*   **Automated Daily Tasks:** Automatically clear stamina, complete daily training, claim rewards, handle assignments, and farm Calyxes.
*   **Automated Weekly Tasks:** Tackle Simulated Universe, and Memory of Chaos runs.
*   **SRGF抽卡记录导出:** Support SRGF standard and **automatic dialogue**.
*   **Notification System:** Receive push notifications for completed daily training and other tasks.
*   **Intelligent Automation:** Automatically starts tasks upon refresh or when stamina reaches a specified level.
*   **Customizable Actions:** Includes sound notifications, automatic game closing, or computer shutdown after tasks are finished.

> Utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for Calyx farming.

For more details, see the [configuration file](assets/config/config.example.yaml) or the graphical interface settings.  🌟 If you like it, please give it a star! |･ω･) 🌟｜QQ Group [Join Now](https://qm.qq.com/q/LpfAkDPlWa) TG Group [Join Now](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![March7th Assistant Interface](assets/screenshot/README.png)

## Important Notes

*   **PC Requirements:**  Requires a PC running the game in a `1920*1080` resolution window or full-screen mode (HDR is not supported).
*   **Simulated Universe Documentation:** [Project Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md)  [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md)
*   **Background Operation:** For background execution or multi-monitor setups, try [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:**  Report any issues in the [Issue](https://github.com/moesnow/March7thAssistant/issues) section and discuss in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions). Pull Requests are welcome! [PR](https://github.com/moesnow/March7thAssistant/pulls)

## Download and Installation

1.  Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest)
2.  Download the latest release.
3.  Extract the downloaded archive.
4.  Double-click `March7th Launcher.exe` (the icon featuring March7th) to launch the graphical interface.

If you need to run the Assistant using the **Task Scheduler** or execute the **complete run** directly, use the terminal icon `March7th Assistant.exe`.

Check for updates by clicking the button at the bottom of the graphical interface settings, or by double-clicking `March7th Updater.exe`.

## Source Code Usage (Advanced)

If you're comfortable with coding, here's how to run from source:

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
<summary>Development Notes</summary>

Crop parameters for capturing screenshots can be obtained using the screenshot capture function in the Assistant's toolbox.

`python main.py` supports arguments like `fight/universe/forgottenhall`.

</details>

---

If you like this project, you can support the author with a coffee! ☕

Your support fuels the development and maintenance of this project!🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant is built with the help of these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Calyx Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

[**Back to Top**](https://github.com/moesnow/March7thAssistant)
```
Key improvements and SEO optimizations:

*   **Clear Title & Hook:** A clear title followed by a concise one-sentence hook at the beginning.  The hook uses keywords related to Honkai: Star Rail and automation to grab attention.
*   **Keywords throughout:** Integrated keywords such as "Honkai: Star Rail", "automation", "daily tasks", "Simulated Universe", "Calyx farming", "SRGF", and "notifications" in headings and key feature descriptions to improve searchability.
*   **Structured Content:** Used clear headings and subheadings (e.g., "Key Features", "Download and Installation", "Important Notes") to organize the information, making it easier to read and understand.
*   **Bulleted Lists:** Used bullet points to highlight the main features, improving readability and making it easier for users to scan and understand the functionality.
*   **Concise Descriptions:** Kept the descriptions brief and to the point, focusing on the most important aspects of each feature.
*   **Call to Action:** The 'Back to Top' link at the end encourages navigation.
*   **GitHub Link:** Included a clear "Back to Top" link at the end.
*   **Optimized for Search Engines:** Included relevant keywords in the title, headings, and descriptions to improve search engine optimization.
*   **Removed Unnecessary Details:** Streamlined the README to focus on the most important information for users.