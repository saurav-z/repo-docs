<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7th Assistant · 三月七小助手
  </h1>
  <p>Automate your Honkai: Star Rail daily tasks with March7th Assistant!</p>
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
  <a href="./README_EN.md">English</a> | <a href="./README_TW.md">繁體中文</a> | 简体中文
  <br/>
  <a href="https://m7a.top/#/assets/docs/Tutorial">Quick Start Tutorial</a> | <a href="https://m7a.top/#/assets/docs/FAQ">FAQ</a>
</div>

## Key Features

March7th Assistant automates various Honkai: Star Rail tasks, saving you time and effort!

*   **Automated Daily Tasks:** Complete daily training, claim rewards, manage commissions, and farm for resources.
*   **Weekly Task Automation:** Automate Simulated Universe, Memory of Chaos, and more.
*   **Automated Task Completion:** Trigger automated task execution upon specified conditions such as time or resource levels.
*   **Automated Task Notifications:** Receive message notifications to monitor your task completion.
*   **Automatic Game Actions:** Auto start the game when tasks are ready, and sound alerts with automatic game shutdown upon completion.
*   **Automated Card Extraction**:  Extract Card Information (SRGF compatible)

> Leverages [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for resource farming.

Customize your experience via the [configuration file](assets/config/config.example.yaml) or the graphical user interface.  🌟 Show your support with a star! 🌟  Join the QQ Group [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) or TG Group [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1).

## Screenshots

![March7th Assistant Interface](assets/screenshot/README.png)

## Important Notes

*   **Compatibility:** Requires a PC with a `1920x1080` resolution or full-screen game mode (HDR not supported).
*   **Simulated Universe:** Refer to the [Auto_Simulated_Universe documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Operation:** For background operation and multi-monitor setups, consider using [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:** Report any issues in the [Issue tracker](https://github.com/moesnow/March7thAssistant/issues) and discuss in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull requests are welcome! [PR](https://github.com/moesnow/March7thAssistant/pulls)

## Download and Installation

Get started quickly! Download the latest release from the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page. Unzip the archive and launch `March7th Launcher.exe` (the March7th icon) to open the GUI.

For scheduled or command-line execution, use `March7th Assistant.exe`.

Check for updates within the GUI or by running `March7th Updater.exe`.

## Running from Source (For Developers)

If you're a developer, follow these steps to run the assistant from source. Otherwise, use the pre-built releases above.

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

You can obtain crop parameters using the screenshot capture feature within the Assistant's toolbox.

Use the following arguments with `python main.py`: `fight/universe/forgottenhall`
</details>

---

Support the project!  Buy the developer a coffee ☕ via WeChat.

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7th Assistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Resource Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Framework: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributing

Contribute to March7th Assistant! Check out the [contributing guidelines](CONTRIBUTING.md).

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors"/>
</a>

## Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

[Back to top](#) - Return to the [March7th Assistant](https://github.com/moesnow/March7thAssistant) repository.
```
Key improvements and SEO considerations:

*   **Clear Title and Introduction:** The title is retained and a concise, engaging introduction is added for immediate user interest.
*   **SEO-Friendly Keywords:** The text incorporates relevant keywords like "Honkai: Star Rail," "automation," "daily tasks," and "game assistant."
*   **Structured Headings:**  Uses clear, descriptive headings (Key Features, Important Notes, Download and Installation, etc.) for better readability and organization.
*   **Bulleted Lists:** Key features are presented in a clear, easy-to-scan bulleted list.
*   **Action-Oriented Language:**  Uses phrases like "Automate your..." and "Get started quickly!" to encourage user action.
*   **Internal Linking:** Includes a "Back to top" link and links back to the GitHub repository, enhancing navigation and SEO.
*   **Concise Summarization:**  The original README is summarized without losing essential information.
*   **Developer Focus:** Separate instructions for running from source are provided and clearly marked to cater to that audience.
*   **Alt Text for Images**:  Provides alt text for the images
*   **Simplified Installation Instructions**: Streamlined the download and install section.
*   **Improved Formatting**: Cleaned up and enhanced the formatting for better readability.
*   **Contributing guidelines**: Added an explicit reference to contributing guidelines.