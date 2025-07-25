<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7thAssistant · 星穹铁道自动化助手
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

Get started quickly with the [Tutorial](https://m7a.top/#/assets/docs/Tutorial) or check the [FAQ](https://m7a.top/#/assets/docs/FAQ) for assistance.
</div>

## Automate Your Honkai: Star Rail Experience with March7thAssistant

March7thAssistant is a powerful Windows automation tool designed to streamline your daily and weekly tasks in Honkai: Star Rail, saving you time and effort.  Check out the [original repository](https://github.com/moesnow/March7thAssistant) for more details!

### Key Features:

*   **Daily Task Automation:** Automates daily tasks such as stamina management, daily training, claiming rewards, dispatching commissions, and "farming" activities.
*   **Weekly Task Automation:**  Automates weekly tasks including Simulated Universe, Forgotten Hall, and Echo of War.
*   **Automated Card Extraction:** Exports your card history with support for the [SRGF](https://uigf.org/zh/standards/SRGF.html) standard. Includes **automated dialogue** functionality.
*   **Notification Support:** Sends notifications about the completion of tasks like daily training.
*   **Automated Triggers:** Automatically starts tasks when daily tasks refresh or stamina reaches a specified level.
*   **Customizable Actions:** Provides sound notifications, auto-close game, or shutdown options after tasks are completed.

**Utilizes:** [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for "farming".

For details, see the [configuration file](assets/config/config.example.yaml) or the graphical interface settings.  🌟 Show your support with a star! |･ω･) 🌟 | Join the QQ group [here](https://qm.qq.com/q/LpfAkDPlWa) or the TG group [here](https://t.me/+ZgH5zpvFS8o0NGI1).

## Screenshots

![Main Interface](assets/screenshot/README.png)

## Important Notes:

*   **PC Requirement:** Requires a Windows PC running the game in a `1920x1080` resolution window or full-screen mode (HDR is not supported).
*   **Simulated Universe Documentation:**  Refer to the [project documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for the Simulated Universe automation.
*   **Background Operation:** For background operation or multi-monitor setups, consider using [remote desktop](https://m7a.top/#/assets/docs/Background).
*   **Issue Reporting:** Report any issues on [GitHub Issues](https://github.com/moesnow/March7thAssistant/issues).  Discuss and ask questions in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  PRs are welcome!
*   **Known Issues:**  Due to the nature of the game, the assistant can encounter errors. Please provide as much detail as possible.

## Download and Installation

1.  Go to [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Download the latest release.
3.  Extract the contents.
4.  Double-click `March7th Launcher.exe` (the March7th icon) to launch the graphical interface.

For scheduled tasks or direct execution of the "complete run" feature, use the terminal icon `March7th Assistant.exe`.

Check for updates using the button at the bottom of the graphical interface or by double-clicking `March7th Updater.exe`.

## Source Code Execution (For Developers & Advanced Users)

If you are new, please use the download and install method above.

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

The cropping coordinates can be obtained using the screenshot capture function in the assistant's toolbox.

`python main.py` supports parameters like `fight`, `universe`, and `forgottenhall`.

</details>

---

If you enjoy this project, you can support the author with a coffee ☕ via WeChat donation:

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on these open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   "Farming" Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors
<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">

  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />

</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)
```
Key improvements and SEO considerations:

*   **Clear Headline:**  Uses the project name directly and keywords like "Honkai: Star Rail," and "Automation" in the main title.
*   **Concise Hook:** Provides a strong, value-driven one-sentence introduction to immediately engage the reader.
*   **Keyword Optimization:**  Incorporates relevant keywords throughout the text, such as "automation," "tasks," "stamina," "rewards," "Simulated Universe," etc.  This helps with search engine rankings.
*   **Structured Content:**  Uses clear headings (Key Features, Important Notes, Download and Installation, etc.) to improve readability and help users quickly find information.
*   **Bulleted Lists:** Highlights key features and requirements using bullet points for easy scanning.
*   **Action-Oriented Language:** Uses phrases like "Automate Your... Experience," "Download," and "Get started" to encourage user engagement.
*   **Updated URLs:** Maintains all URLs to relevant sites.
*   **Clear Call to Actions:** Encourages users to use features, check the FAQ, and contribute (star the repo, report issues, PRs)
*   **Concise Summaries:** Provides brief but informative summaries of each feature.
*   **Simplified Instructions:** Download and Installation instructions are simplified and clearer.
*   **SEO-Friendly Description:**  The initial paragraphs are rewritten to be more search engine-friendly and give more context to users.
*   **Included Links:** Maintain all useful links to the tutorials, FAQ, etc.
*   **Code Formatting:** Kept existing code snippets, but made formatting more readable.