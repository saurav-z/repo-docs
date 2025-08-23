<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <p><em>Automate your daily tasks in Honkai: Star Rail with March7thAssistant!</em></p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release (latest by date)" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub All Releases Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">

**简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)

Get started quickly: [Tutorial](https://m7a.top/#/assets/docs/Tutorial)

Find answers in the [FAQ](https://m7a.top/#/assets/docs/FAQ)

</div>

## Key Features

March7thAssistant is a powerful automation tool for Honkai: Star Rail, offering a range of features to enhance your gameplay.

*   **Automated Daily Tasks:** Automates daily activities such as stamina claiming, daily training, reward collection, commissions, and farming.
*   **Automated Weekly Tasks:** Automates weekly tasks like Simulated Universe, Echo of War, and Forgotten Hall.
*   **Automated Card Draw Record Export**: Support [SRGF](https://uigf.org/zh/standards/SRGF.html) standard, and **Automatic conversation**.
*   **Notification Support**: Provides message push notifications for the completion of daily training and other tasks.
*   **Automated Triggers**: Automatically starts tasks when daily tasks are refreshed or stamina is restored to a specified value.
*   **Customizable Actions**: Includes sound notifications upon task completion, with options to automatically close the game or shut down your PC.

> This project leverages the [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) project for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for farming automation.

Find detailed configuration instructions in the [configuration file](assets/config/config.example.yaml) or through the graphical user interface.  🌟 Star the repo if you like it! |･ω･) 🌟｜QQ Group [Click to Join](https://qm.qq.com/q/LpfAkDPlWa)｜TG Group [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![March7thAssistant in Action](assets/screenshot/README.png)

## Important Notes

*   Requires a **PC** running the game in a `1920x1080` resolution window or full-screen mode (HDR is not supported).
*   For Simulated Universe related information, see the [project documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   For background operation or multi-monitor setups, consider using [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   Report any issues in the [Issue tracker](https://github.com/moesnow/March7thAssistant/issues) and discuss in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull Requests are welcome! [PRs](https://github.com/moesnow/March7thAssistant/pulls).

## Download and Installation

1.  Download the latest release from [Releases](https://github.com/moesnow/March7thAssistant/releases/latest).
2.  Extract the downloaded archive.
3.  Double-click `March7th Launcher.exe` (the icon with the March7th logo) to launch the graphical interface.

To schedule tasks with the **Task Scheduler** or run the **full program**, use `March7th Assistant.exe` (terminal icon).

Check for updates by clicking the button at the bottom of the graphical interface settings, or by double-clicking `March7th Updater.exe`.

## Running from Source (Advanced Users)

If you are familiar with Python and want to contribute or customize March7thAssistant, follow these steps:

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

You can obtain the crop parameter crop coordinate by using the screenshot capture function in the auxiliary toolbox.

The `python main.py` command also supports command-line arguments: `fight`, `universe`, `forgottenhall`, etc.
</details>

---

If you find this project helpful, consider supporting the developer with a coffee! ☕

Your support motivates the development and maintenance of this project! 🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR Text Recognition: [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributing

We welcome contributions!  Please see the [Contributing Guidelines](CONTRIBUTING.md) for more information.

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors" />
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

---
**[Back to Top](#top)** (link to top of the page)
```

Key improvements and explanations:

*   **SEO-Optimized Title & Description:**  Uses keywords like "Honkai: Star Rail," "automation," and task names to improve searchability.  The one-sentence hook grabs attention.
*   **Clear Headings:**  Uses Markdown headings to organize content (e.g., "Key Features," "Download and Installation").  Added "Contributing" and "Back to Top" sections.
*   **Bulleted Key Features:** Makes the core functionality immediately clear and easy to scan. Uses clear, concise language.
*   **Concise Language:**  Removed unnecessary words and phrases.  Simplified instructions.
*   **Clear Call to Actions:** Encourages users to star the repo, join the QQ/TG groups, and contribute.  Includes a link to the top of the document.
*   **Complete Information:**  Retains all the original information but presents it more effectively.
*   **Contributor Section:**  Adds a section for contributors with a visual image.
*   **Consistent Formatting:**  Ensures consistent formatting throughout the document.
*   **Source Code Instructions:** Kept the source code execution instructions for more advanced users.
*   **Back to the Top Link:** Provides a link to the top of the document for ease of navigation.
*   **Alt Text for Images:** Added descriptive alt text for all images, which helps with accessibility and SEO.
*   **Uses "Automated" to enhance SEO:** Uses the term "automated" to improve search visibility.