# Mobile-Use: Automate Your Phone with Natural Language 🤖

**Tired of tapping and swiping? Mobile-Use lets you control your Android or iOS device with simple, natural language commands.** Explore the future of mobile interaction with this open-source AI agent!  [View the source on GitHub](https://github.com/minitap-ai/mobile-use).

<div align="center">
  <img src="./doc/linkedin-demo-with-text.gif" alt="Mobile-Use in Action" width="70%" />
</div>

<div align="center">
  <a href="https://discord.gg/6nSqmQ9pQs">
    <img src="https://img.shields.io/discord/1403058278342201394?color=7289DA&label=Discord&logo=discord&logoColor=white&style=for-the-badge" alt="Discord">
  </a>
  <a href="https://github.com/minitap-ai/mobile-use/stargazers">
    <img src="https://img.shields.io/github/stars/minitap-ai/mobile-use?style=for-the-badge&color=e0a8dd" alt="GitHub stars">
  </a>
  <p align="center">
    <a href="https://discord.gg/6nSqmQ9pQs"><b>Join the Discord Community</b></a> •
    <a href="https://x.com/minitap_ai?t=iRWtI497UhRGLeCKYQekig&s=09"><b>Follow us on X/Twitter</b></a>
  </p>
</div>

Mobile-use is an open-source AI agent that empowers you to control your Android or iOS device effortlessly.  It understands natural language commands, allowing you to automate tasks, extract data, and navigate apps with ease.

>  Your feedback drives Mobile-use's evolution!  Join the conversation on [Discord](https://discord.gg/6nSqmQ9pQs) or contribute directly; all contributions are welcomed!

## 🚀 Key Features

*   🗣️ **Natural Language Control:**  Interact with your phone using plain, everyday language.
*   📱 **UI-Aware Automation:**  Intelligently navigates and interacts with app interfaces.
*   📊 **Data Scraping:** Extract information from any app and structure it into your desired format (e.g., JSON) using natural language.
*   🔧 **Extensible & Customizable:** Easily integrate and configure different Large Language Models (LLMs) to power your mobile-use agent.

## 🏆 Benchmarks

Mobile-use is the **#1 open-source project** on the AndroidWorld benchmark, showcasing its powerful capabilities.

<p align="center">
  <img src="./doc/benchmark.png" alt="Benchmark results" />
</p>

Find more details at: [https://minitap.ai/research/mobile-ai-agents-benchmark](https://minitap.ai/research/mobile-ai-agents-benchmark)

View the official leaderboard: [https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0)

## 🚦 Getting Started

Ready to experience the future of mobile interaction?  Follow these steps to get started:

1.  **Set up Environment Variables:**
    *   Copy the example `.env.example` file to `.env` and add your required API keys.  At least an OpenAI key is necessary.
    ```bash
    cp .env.example .env
    ```

2.  **(Optional) Customize LLM Configuration:**
    *   To use different models or providers, create your own LLM configuration file.
        ```bash
        cp llm-config.override.template.jsonc llm-config.override.jsonc
        ```
    *   Edit `llm-config.override.jsonc` to match your specific requirements.

### Quick Launch (Docker - Android Only)

> [!NOTE]
> This quickstart is currently only available for Android devices/emulators, and requires Docker to be installed.

**Prerequisites:**

*   Plug in your Android device and enable USB debugging via Developer Options, **OR** launch an Android emulator.

> [!IMPORTANT]
> At some point, the terminal will pause, and Maestro will request anonymous usage data collection. Accept (enter 'Y') or decline (enter 'n') as desired.

**Run these commands in your terminal:**

1.  **For Linux/macOS:**
    ```bash
    chmod +x mobile-use.sh
    ./mobile-use.sh \
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" \
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

2.  **For Windows (Powershell):**
    ```powershell
    powershell.exe -ExecutionPolicy Bypass -File mobile-use.ps1 `
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" `
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

> [!NOTE]
> If using your own device, accept the ADB connection requests that appear.  Maestro will also install its APK, which will require installation approval.

#### 🛠️ Troubleshooting Docker

The script attempts to connect to your device via IP.

*   **Ensure your device is on the same Wi-Fi network as your computer.**

##### 1. Device IP Not Found

If the script fails with:
```
Could not get device IP. Is a device connected via USB and on the same Wi-Fi network?
```
You may need to determine your phone's WLAN interface name using `adb shell ip addr show up`.  Then, add the `--interface <YOUR_INTERFACE_NAME>` option to the script.

##### 2. Docker Connection Issues

A firewall may be blocking the connection to `<DEVICE_IP>:5555` inside Docker. There may not be a simple fix for this.

##### 3. `ghcr.io` Docker Image Pull Failure (Unauthorized)

If you've previously used `ghcr.io` for private repositories and encounter this issue, try:
```bash
docker logout ghcr.io
```
Then rerun the script.

### Manual Launch (Development Mode)

For developers:

#### 1. Device Support

Mobile-use currently supports:

*   **Physical Android Phones:** Connect via USB with USB debugging enabled.
*   **Android Simulators:** Set up through Android Studio.
*   **iOS Simulators:** Supported for macOS users.

> [!NOTE]
> Physical iOS devices are not yet supported.

#### 2. Prerequisites

**Android:**

*   **[Android Debug Bridge (ADB)](https://developer.android.com/studio/releases/platform-tools)**: For connecting to your device.

**iOS:**

*   **[Xcode](https://developer.apple.com/xcode/)**:  Apple's IDE for iOS development.

**Required Installations:**

*   **[uv](https://github.com/astral-sh/uv)**: A fast Python package manager.
*   **[Maestro](https://maestro.mobile.dev/getting-started/installing-maestro)**: The framework to interact with your device.

#### 3. Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/minitap-ai/mobile-use.git && cd mobile-use
    ```

2.  [**Setup environment variables**](#-getting-started)

3.  **Create and Activate Virtual Environment:**
    ```bash
    # Creates a .venv directory using the Python version specified in .python-version
    uv venv

    # Activate the environment:
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    # Sync with the locked dependencies for consistency
    uv sync
    ```

## ⌨️ Usage

Run Mobile-use by passing your command as an argument:

**Example 1: Basic Command**
```bash
python ./src/mobile_use/main.py "Go to settings and tell me my current battery level"
```

**Example 2: Data Scraping**

Extract and format data (e.g., get a list of unread emails):
```bash
python ./src/mobile_use/main.py \
  "Open Gmail, find all unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

> [!NOTE]
> If you haven't configured a specific model, Mobile-use will prompt you to choose from the available options.

## 🤝 Contributing

We welcome contributions! Review our **[Contributing Guidelines](CONTRIBUTING.md)** to get involved.

## 📈 Star History

<p align="center">
  <a href="https://star-history.com/#minitap-ai/mobile-use&Date">
    <img src="https://api.star-history.com/svg?repos=minitap-ai/mobile-use&type=Date" alt="Star History Chart" />
  </a>
</p>

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.