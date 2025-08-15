<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p>Automate your daily Honkai: Star Rail tasks with this helpful assistant!</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Release Version" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
  <br>
  <a href="https://github.com/moesnow/March7thAssistant">
    <img src="https://img.shields.io/badge/View%20on%20GitHub-blue?style=flat-square&logo=github" alt="View on GitHub"/>
  </a>
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

Need help? Check out the [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Key Features

March7th Assistant automates various daily and weekly tasks, providing a smoother Honkai: Star Rail experience. Here's what you can do:

*   **Daily Tasks Automation:** Clear stamina, complete daily training, claim rewards, handle commissions, and farm Calyxes (锄大地).
*   **Weekly Tasks Automation:** Conquer the Echo of War, Simulated Universe, and Forgotten Hall.
*   **Automated Task Completion:** Configure automatic launches and shutdowns based on task completion or stamina recovery.
*   **Customizable Notifications:** Receive notifications for daily training completion and other events.
*   **Import/Export:** Supports [SRGF](https://uigf.org/zh/standards/SRGF.html) standard for exporting and importing gacha records.
*   **Automated Dialog:** Supports automatic dialogue during the gacha record import process.

> This project utilizes [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for Calyx farming.

For detailed configuration options, see the [configuration file](assets/config/config.example.yaml) or configure through the graphical interface.  🌟 If you like this project, please give it a star! |･ω･) 🌟 |  QQ Group [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) TG Group [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![Screenshot of March7th Assistant](assets/screenshot/README.png)

## Important Notes

*   **Supported Platform:** Requires a Windows PC with a `1920*1080` resolution, running the game in windowed mode or full screen (HDR is not supported).
*   **Simulated Universe Documentation:** Refer to [Auto_Simulated_Universe Documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for more details.
*   **Background Operation:** For background operation or multi-monitor setups, consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Report any bugs or issues on the [Issue Tracker](https://github.com/moesnow/March7thAssistant/issues).  For discussions, visit the [Discussions](https://github.com/moesnow/March7thAssistant/discussions) page.  PRs are welcome on the [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls) page.

## Installation and Usage

1.  **Download:** Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page and download the latest release.
2.  **Run:** Extract the downloaded archive and double-click the `March7th Launcher.exe` (icon with March 7th image) to open the graphical interface.
3.  **Advanced:** If you need to schedule tasks or run the assistant directly, use the `March7th Assistant.exe` (terminal icon).
4.  **Update:** Check for updates by clicking the button at the bottom of the graphical interface settings or by double-clicking the `March7th Updater.exe`.

## Source Code Execution (For Developers/Advanced Users)

*If you're a beginner, the steps above are sufficient.  Skip this section.*

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

The crop parameters for screen capture can be obtained using the screen capture functionality within the assistant's toolbox.

`python main.py` supports the following parameters: `fight/universe/forgottenhall`
</details>

---

If you enjoy this project, you can support the developer with a coffee! ☕

Your support fuels the development and maintenance of this project! 🚀

![Sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant relies on these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Calyx Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors"/>
</a>

## Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:** The title is emphasized and a one-sentence hook is added to grab attention.
*   **Concise and Keyword-Rich Headings:** Using headings like "Key Features" and "Installation and Usage" helps with readability and SEO.
*   **Bulleted Key Features:**  Makes the features easily scannable and highlights important keywords.
*   **GitHub Link:** Added a direct link to the GitHub repository.
*   **Alt Text for Images:** Added descriptive alt text to all images, for accessibility and SEO.
*   **Contextual Keywords:** Using terms relevant to the game "Honkai: Star Rail", "automation", and other relevant keywords.
*   **Action-Oriented Language:** Using verbs like "Automate", "Conquer" and "Claim" to keep things dynamic.
*   **Clear Installation Instructions:** Simplified and improved installation section.
*   **Comprehensive Summary:** The README is now better organized and easier to understand.
*   **Removed redundant information**