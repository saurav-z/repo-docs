# Mobile-Use: Automate Your Mobile with Natural Language 📱🤖

**Control your Android or iOS device with simple, natural language commands using Mobile-Use, an open-source AI agent.**

<div align="center">
    <img src="./doc/linkedin-demo-with-text.gif" alt="Mobile-Use in Action" width="600"/>
</div>

<div align="center">
    <a href="https://discord.gg/6nSqmQ9pQs"><img src="https://img.shields.io/discord/1403058278342201394?color=7289DA&label=Discord&logo=discord&logoColor=white&style=for-the-badge" alt="Discord Badge"/></a>
    <a href="https://github.com/minitap-ai/mobile-use/stargazers"><img src="https://img.shields.io/github/stars/minitap-ai/mobile-use?style=for-the-badge&color=e0a8dd" alt="GitHub stars"/></a>
    <p align="center">
        <a href="https://discord.gg/6nSqmQ9pQs"><b>Join our Discord</b></a> |
        <a href="https://x.com/minitap_ai?t=iRWtI497UhRGLeCKYQekig&s=09"><b>Follow us on X (Twitter)</b></a>
        <br>
        <a href="https://github.com/minitap-ai/mobile-use"><b>View on GitHub</b></a>
    </p>
</div>

Mobile-Use empowers you to control your mobile device using natural language. This open-source AI agent understands your commands and interacts with your phone's UI, enabling automation of tasks from sending messages to navigating complex apps.

> _Your feedback is crucial! Join the conversation on [Discord](https://discord.gg/6nSqmQ9pQs) or contribute directly to help shape the future of Mobile-Use._ ❤️

## ✨ Key Features

*   🗣️ **Natural Language Control:** Control your phone using everyday language.
*   📱 **UI-Aware Automation:** Intelligently navigate and interact with app interfaces.
*   📊 **Data Scraping:** Extract and structure information from any app into formats like JSON using natural language descriptions.
*   🔧 **Extensible & Customizable:** Easily configure different LLMs to power the agents behind Mobile-Use.

## 🏆 Benchmarks

Mobile-Use is the **#1 open-source project on the AndroidWorld benchmark** for mobile AI agents.

<p align="center">
  <img src="./doc/benchmark.png" alt="AndroidWorld Benchmark" width="600"/>
</p>

*   **Learn More:** [Mobile AI Agents Benchmark](https://minitap.ai/research/mobile-ai-agents-benchmark)
*   **Leaderboard:** [Official Benchmark](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0)

## 🚀 Getting Started

Ready to automate your mobile experience? Follow these steps:

1.  **Set up Environment Variables:**

    *   Copy the example `.env.example` file to `.env` and add your API keys.  An OpenAI key is required.

    ```bash
    cp .env.example .env
    ```

2.  **(Optional) Customize LLM Configuration:**

    *   Create your own LLM configuration file.
    ```bash
    cp llm-config.override.template.jsonc llm-config.override.jsonc
    ```
    *   Then, edit `llm-config.override.jsonc` to fit your needs.

### Quick Launch (Docker - Android Only)

> [!NOTE]
> This quickstart is currently only available for Android devices/emulators, and requires Docker.

1.  **Prepare your Android Device/Emulator:**
    *   Connect your Android device and enable USB debugging in Developer Options.
    *   Or launch an Android emulator.

> [!IMPORTANT]
> Maestro may prompt you to collect anonymous usage data.  You can choose to accept ('Y') or decline ('n').

2.  **Run the Script:**

    *   **Linux/macOS:**

    ```bash
    chmod +x mobile-use.sh
    ./mobile-use.sh \
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" \
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

    *   **Windows (Powershell):**

    ```powershell
    powershell.exe -ExecutionPolicy Bypass -File mobile-use.ps1 `
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" `
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

> [!NOTE]
> Accept ADB and Maestro installation requests on your device if prompted.

#### 🧰 Troubleshooting

*   **Device IP Not Found:** Ensure your device is on the same Wi-Fi network and determine your phone's WLAN interface using `adb shell ip addr show up`. Add `--interface <YOUR_INTERFACE_NAME>` to the script.

*   **Docker Connection Issues:** Firewall issues may prevent connections.

*   **GHCR Docker Image Errors:** Run `docker logout ghcr.io` if you've used `ghcr.io` for private repositories previously.

### Manual Launch (Development Mode)

For developers who want to set up the environment manually:

#### 1. Device Support

*   **Physical Android Phones:** Supported with USB debugging enabled.
*   **Android Simulators:** Supported via Android Studio.
*   **iOS Simulators:** Supported for macOS users.

> [!NOTE]
> Physical iOS devices are not yet supported.

#### 2. Prerequisites

*   **Android:**
    *   [Android Debug Bridge (ADB)](https://developer.android.com/studio/releases/platform-tools)
*   **iOS:**
    *   [Xcode](https://developer.apple.com/xcode/)
*   **General:**
    *   [uv](https://github.com/astral-sh/uv) - Python package manager.
    *   [Maestro](https://maestro.mobile.dev/getting-started/installing-maestro) - For device interaction.

#### 3. Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/minitap-ai/mobile-use.git && cd mobile-use
    ```

2.  [**Setup environment variables**](#-getting-started)

3.  **Create & Activate Virtual Environment:**

    ```bash
    # This will create a .venv directory using the Python version in .python-version
    uv venv

    # Activate the environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate
    ```

4.  **Install Dependencies:**

    ```bash
    # Sync with the locked dependencies for a consistent setup
    uv sync
    ```

## 👨‍💻 Usage

Run Mobile-Use by passing your command as an argument:

**Example 1: Basic Command**

```bash
python ./src/mobile_use/main.py "Go to settings and tell me my current battery level"
```

**Example 2: Data Scraping**

Extract specific information in a structured format:

```bash
python ./src/mobile_use/main.py \
  "Open Gmail, find all unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

> [!NOTE]
> If a specific model isn't configured, Mobile-Use will prompt you to select one.

## ❤️ Contributing

We welcome contributions! Review our **[Contributing Guidelines](CONTRIBUTING.md)** to get started.

## ⭐ Star History

<p align="center">
  <a href="https://star-history.com/#minitap-ai/mobile-use&Date">
    <img src="https://api.star-history.com/svg?repos=minitap-ai/mobile-use&type=Date" alt="Star History Chart" />
  </a>
</p>

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.