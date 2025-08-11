<div align="center">
  <h1 align="center">
    <img src="./assets/screenshot/March7th.png" width="200" alt="March7th Assistant Logo">
    <br/>
    March7thAssistant · 三月七小助手
  </h1>
  <p>Automate your daily Honkai: Star Rail tasks with this powerful assistant!</p>
  <a href="https://trendshift.io/repositories/3892" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3892" alt="moesnow%2FMarch7thAssistant | Trendshift" style="width: 200px; height: 46px;" width="250" height="46"/></a>
</div>

<br/>

<div align="center">
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows-blue?style=flat-square&color=4096d8" />
  <img alt="GitHub Release" src="https://img.shields.io/github/v/release/moesnow/March7thAssistant?style=flat-square&color=f18cb9" />
  <img alt="GitHub Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
  <a href="./README_EN.md">English</a> | <a href="./README_TW.md">繁體中文</a> | 简体中文
  <br/>
  Get started quickly: <a href="https://m7a.top/#/assets/docs/Tutorial">Tutorial</a>
  <br/>
  Need help? Check out the <a href="https://m7a.top/#/assets/docs/FAQ">FAQ</a>
</div>

## Key Features

March7thAssistant streamlines your Honkai: Star Rail gameplay with the following features:

*   **Automated Daily Tasks:**  Effortlessly complete daily training, stamina management, reward collection, assignments, and farming.
*   **Weekly Content Support:**  Automate weekly challenges like Simulated Universe and Memory of Chaos.
*   **抽卡记录 Export:** Supports SRGF standards for convenient record-keeping and **automatic dialogue**.
*   **Customizable Notifications:** Receive **message push notifications** upon completion of daily tasks.
*   **Automated Triggers:**  Automatically start tasks upon mission refresh or stamina restoration to a specified value.
*   **Post-Task Actions:**  Configurable actions upon task completion, including **sound notifications, automatic game closing, or system shutdown**.

>   This project leverages the following: [Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe) for Simulated Universe automation and [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail) for farming automation.

For detailed configuration options, see the [configuration file](assets/config/config.example.yaml) or use the graphical user interface.  🌟  If you like this project, please give it a star! |･ω･) 🌟  Join the QQ group [Click to Join](https://qm.qq.com/q/LpfAkDPlWa) or TG group [Click to Join](https://t.me/+ZgH5zpvFS8o0NGI1)

## Screenshots

![March7th Assistant in Action](assets/screenshot/README.png)

## Important Notes

*   **PC Requirements:**  Requires a Windows PC with a `1920*1080` resolution and the game running in windowed or full-screen mode (HDR is not supported).
*   **Simulated Universe:**  Refer to the [project documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md) for the Simulated Universe component.
*   **Background Operation:**  For background operation or multi-monitor setups, consider using [Remote Local Multi-User Desktop](https://m7a.top/#/assets/docs/Background).
*   **Reporting Issues:**  Report any errors via [Issues](https://github.com/moesnow/March7thAssistant/issues) and discuss in the [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  Pull Requests (PRs) are welcome!

## Installation and Usage

1.  **Download:** Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page and download the latest release.
2.  **Launch:** Extract the downloaded archive and double-click `March7th Launcher.exe` (the icon featuring March 7th) to open the graphical interface.
3.  **Automated Runs:** To schedule tasks or run the assistant directly via the terminal, use `March7th Assistant.exe`.
4.  **Updates:** Check for updates using the button in the graphical interface, or by double-clicking `March7th Updater.exe`.

## Running from Source (For Developers)

If you're comfortable with development, follow these steps:

```cmd
# Installation (venv is recommended)
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

You can get the crop parameters for image cropping using the capture screenshot feature in the assistant's toolbox.

The `python main.py` command supports arguments such as `fight`, `universe`, and `forgottenhall`.

</details>

---

If you find this project helpful, consider supporting the developer with a coffee via WeChat! ☕

Your support fuels the development and maintenance of this project!🚀

![sponsor](assets/app/images/sponsor.jpg)

---

## Dependencies and Related Projects

March7thAssistant relies on the following open-source projects:

*   Simulated Universe Automation: [https://github.com/CHNZYX/Auto_Simulated_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   Farming Automation: [https://github.com/linruowuyin/Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   OCR (Optical Character Recognition): [https://github.com/hiroi-sora/PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   GUI Component Library: [https://github.com/zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" alt="Contributors" />
</a>

## Star History

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)

---

**[Back to Top](#)**  (link to the top of the page)
```
Key improvements and SEO considerations:

*   **Clear and Concise Title:** The title includes the project name and a helpful tagline.
*   **SEO-Optimized Description:**  The one-sentence hook is clear and uses relevant keywords: "Automate your daily Honkai: Star Rail tasks with this powerful assistant!"
*   **Keyword Integration:**  Uses keywords like "Honkai: Star Rail," "automation," "daily tasks," "Simulated Universe," and "farming" naturally throughout the README.
*   **Structured Content:**  Uses headings, bullet points, and bold text for easy readability and SEO.
*   **Contextual Links:** Links are provided where needed (e.g., FAQ, Tutorial), improving usability and SEO.
*   **Alt Text for Images:**  Adds `alt` text to images for accessibility and SEO.
*   **Concise Summary:** The key features are presented clearly and concisely.
*   **Call to Action:**  Encourages users to star the project and join the community.
*   **Clear Instructions:**  Installation and usage instructions are easy to follow.
*   **Contributor Section:** A visually appealing contributor section is included.
*   **Star History Chart:**  Adds a visual representation of the project's popularity over time.
*   **Back to Top link** Helps user navigability.