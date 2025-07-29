<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200">
    <br/>
    March7th Assistant - Automate Your Honkai: Star Rail Daily Tasks
  </h1>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
  <br/>
  <a href="https://github.com/moesnow/March7thAssistant">View on GitHub</a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="Latest Release" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
  [简体中文](README.md) | [繁體中文](./README_TW.md) | [English](./README_EN.md)
</div>

## About March7th Assistant

March7th Assistant is a powerful Windows-based automation tool designed to streamline your Honkai: Star Rail gameplay, saving you time and effort on daily tasks.

## Key Features

*   **Daily Task Automation:** Automates daily activities like stamina refills, daily training, reward claiming, assignments, and "Overworld" farming.
*   **Weekly Task Automation:** Handles weekly tasks such as Simulated Universe and Forgotten Hall.
*   **Automated Combat:** Automated farming through the use of [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail).
*   **Automated Simulated Universe:** Fully automate Simulated Universe content using the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) integration.
*   **Automated Draw Record Export:** Supports [SRGF](https://uigf.org/zh/standards/SRGF.html) standard and **Automatic Dialogue.**
*   **Notification System:** Receives notifications for completed daily training and other tasks.
*   **Automated Launch:** Automatically launches tasks upon daily refresh or when stamina reaches a specified value.
*   **Customizable Actions:** Customizable actions for completing tasks like sound notifications, automatic game closure, and shutdown.

## Screenshots

![March7th Assistant Interface](assets/screenshot/README.png)

## Important Notes

*   **Resolution:** Requires a PC with a `1920x1080` resolution window or full-screen mode (HDR not supported).
*   **Simulated Universe:** Refer to the [Auto_Simulated_Universe documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for related information.
*   **Background Usage:** For background operation or multi-monitor setups, consider using the [remote local multi-user desktop](https://m7a.top/#/assets/docs/Background).
*   **Feedback:** Report any issues on the [Issue Tracker](https://github.com/moesnow/March7thAssistant/issues) and discuss them on the [Discussions](https://github.com/moesnow/March7thAssistant/discussions). Contributions are welcome via [Pull Requests](https://github.com/moesnow/March7thAssistant/pulls).

## Installation

1.  Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Extract the archive.
3.  Double-click `March7th Launcher.exe` (with the March7th icon) to open the GUI.

To schedule tasks or execute a "full run," use `March7th Assistant.exe` via the terminal.

To check for updates, use the button at the bottom of the GUI settings or double-click `March7th Updater.exe`.

## Running from Source (For Developers)

If you are not a developer or familiar with the command line, skip this section and use the installation instructions above.

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

You can get the crop parameters by capturing a screenshot within the toolbox.

`python main.py` accepts arguments: `fight/universe/forgottenhall`
</details>

---

If you like this project, you can support the author with a donation (WeChat). Your support will help with the development and maintenance of the project.

![sponsor](assets/app/images/sponsor.jpg)

---

## Dependencies

March7th Assistant leverages the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Overworld Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />
</a>

## Stargazers Over Time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)
```
Key improvements and SEO optimizations:

*   **Clear Headline and Description:**  A concise description immediately stating what the project does and for what game, crucial for search engine optimization.
*   **Strategic Keyword Placement:**  Includes keywords like "Honkai Star Rail," "automation," "daily tasks," and related terms naturally throughout the text.
*   **Bulleted Feature List:**  Uses bullet points for easy readability and emphasizes key features. This is good for both users and SEO.
*   **Meaningful Headings:** Organizes information with relevant headings for clarity and SEO.
*   **Strong Call to Action/Engagement:** Provides clear calls to action, such as a "View on GitHub" link prominently displayed at the top.
*   **Concise language**: Avoids unnecessary fluff, getting straight to the value proposition.
*   **Improved formatting**: Uses Markdown for proper formatting, including bolding of key terms and use of code blocks.
*   **SEO-Friendly Title & Description:** The title includes a keyword "March7th Assistant" and the one-sentence hook provides an introductory sentence that uses relevant keywords to let the user know immediately what the project is.
*   **Internal Linking**: Added a prominent link to the GitHub repository.