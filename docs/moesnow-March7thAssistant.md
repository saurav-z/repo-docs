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
  <img alt="Total Downloads" src="https://img.shields.io/github/downloads/moesnow/March7thAssistant/total?style=flat-square&color=4096d8" />
</div>

<br/>

<div align="center">
  **简体中文** | [繁體中文](./README_TW.md) | [English](./README_EN.md)
</div>

## Automate Your Honkai: Star Rail Daily Grind with March7thAssistant

March7thAssistant is a powerful automation tool designed to streamline your daily and weekly tasks in Honkai: Star Rail, saving you time and effort.  Check out the [original repository](https://github.com/moesnow/March7thAssistant) for more information.

## Key Features

*   **Automated Daily Tasks**:
    *   Automate daily tasks like clearing stamina, daily training, claiming rewards, and completing commissions.
    *   Includes support for "锄大地" (farming) tasks.
*   **Weekly Content Automation**:
    *   Automates weekly tasks like Simulated Universe and Forgotten Hall.
*   **Automated Card Draw Export**:
    *   Supports SRGF standard for exporting card draw records.
    *   Includes **automatic dialog interaction**.
*   **Notifications & Triggers**:
    *   Receive **message push notifications** upon completion of daily tasks.
    *   **Automatic launch** upon task refresh or stamina reaching a specified value.
*   **Post-Task Actions**:
    *   Offers **sound notifications** upon task completion.
    *   Can be configured to **automatically close the game or shut down your computer** after tasks are done.

## How to Get Started

1.  **Download**: Go to the [Releases](https://github.com/moesnow/March7thAssistant/releases/latest) page.
2.  **Extract**: Download and extract the latest release.
3.  **Launch**: Double-click `March7th Launcher.exe` (with the March7th icon) to launch the graphical interface.

For scheduled tasks or direct execution, use `March7th Assistant.exe`.  To check for updates, click the button at the bottom of the graphical interface or double-click `March7th Updater.exe`.

## Important Notes

*   **Resolution**: Requires a PC with a `1920x1080` resolution window or full-screen mode (HDR is not supported).
*   **Simulated Universe**: For related information, consult the [Auto_Simulated_Universe documentation](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/index.md) and [Q&A](https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/qa.md).
*   **Background Running**: For background operation or multiple monitors, explore [Remote Desktop](https://m7a.top/#/assets/docs/Background).
*   **Support**: Report issues via [Issues](https://github.com/moesnow/March7thAssistant/issues) and discuss them in [Discussions](https://github.com/moesnow/March7thAssistant/discussions).  You're welcome to submit [PRs](https://github.com/moesnow/March7thAssistant/pulls).

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute.

## Development (For Developers)

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

You can obtain crop parameters for image capturing using the capture screenshot feature in the assistant toolbox.

You can also run python main.py with arguments, such as fight/universe/forgottenhall.

</details>

---

## Support the Project

If you find this project helpful, you can buy the author a coffee ☕!  Your support fuels the ongoing development and maintenance of this project!

![sponsor](assets/app/images/sponsor.jpg)

---

## Related Projects

March7thAssistant leverages these great open-source projects:

*   **Simulated Universe Automation**: [Auto\_Simulated\_Universe](https://github.com/CHNZYX/Auto_Simulated_Universe)
*   **Farming Automation**: [Fhoe-Rail](https://github.com/linruowuyin/Fhoe-Rail)
*   **OCR**: [PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json)
*   **GUI Library**: [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

## Contributors

<a href="https://github.com/moesnow/March7thAssistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=moesnow/March7thAssistant" />
</a>

## Stargazers over time

[![Star History](https://starchart.cc/moesnow/March7thAssistant.svg?variant=adaptive)](https://starchart.cc/moesnow/March7thAssistant)
```
Key improvements and explanations:

*   **SEO Optimization**:  Uses relevant keywords like "Honkai Star Rail", "Automation", "Daily tasks", and "Automate".
*   **Clear Headings**: Uses proper markdown headings for better organization and readability.
*   **Concise Summary**: Starts with a clear, benefit-driven one-sentence hook.
*   **Bulleted Key Features**:  Uses bullet points to highlight important features, making them easy to scan.
*   **Actionable Steps**:  Provides clear, step-by-step instructions on how to get started.
*   **Emphasis on Benefits**: Highlights what the user gains from using the tool (saves time, reduces effort).
*   **Community Focused**: Encourages contributions and provides links for support.
*   **Clear Language**: Uses straightforward language.
*   **Complete, Concise, and Accurate**: Retains all important information from the original README while improving its structure and clarity.
*   **Focus on User Experience**:  Prioritizes user-friendliness and ease of understanding.
*   **Uses Alt text on Images**: More accessible and SEO-friendly.
*   **Clearer Call to Action**: The revised README provides a stronger call to action, prompting users to explore the project.
*   **Title optimization**: Added more keywords like "星穹铁道自动化助手".