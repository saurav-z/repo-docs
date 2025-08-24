# Mobile-Use: Automate Your Phone with Natural Language 📱🤖

> Take control of your Android or iOS device with natural language using **Mobile-Use**, an open-source AI agent. [Explore the code on GitHub](https://github.com/minitap-ai/mobile-use).

<div align="center">
  <img src="./doc/linkedin-demo-with-text.gif" alt="Mobile-Use in Action" width="70%">
</div>

<div align="center">
  <a href="https://discord.gg/6nSqmQ9pQs">
    <img src="https://img.shields.io/discord/1403058278342201394?color=7289DA&label=Discord&logo=discord&logoColor=white&style=for-the-badge" alt="Discord">
  </a>
  <a href="https://github.com/minitap-ai/mobile-use/stargazers">
    <img src="https://img.shields.io/github/stars/minitap-ai/mobile-use?style=for-the-badge&color=e0a8dd" alt="GitHub stars">
  </a>
  <p align="center">
    <a href="https://discord.gg/6nSqmQ9pQs"><b>Join the Discord Community</b></a> |
    <a href="https://x.com/minitap_ai?t=iRWtI497UhRGLeCKYQekig&s=09"><b>Follow on X (Twitter)</b></a>
  </p>
</div>

Mobile-Use empowers you to control your smartphone using simple, natural language commands. This innovative AI agent understands your requests and automates tasks within your Android or iOS device. From sending messages and navigating apps to extracting data, Mobile-Use simplifies your mobile experience.

**Key Features:**

*   🗣️ **Natural Language Control:** Command your phone using everyday language.
*   📱 **UI-Aware Automation:** Intelligent navigation through app interfaces.
*   📊 **Data Scraping:** Extract and structure information from any app into your desired format (e.g., JSON).
*   🔧 **Extensible & Customizable:** Easily configure different LLMs to power the agents.

## Benchmarks & Performance

Mobile-Use is a leader in the field, achieving the #1 spot in the open-source pass@1 on the AndroidWorld benchmark.

<div align="center">
  <img src="./doc/benchmark.png" alt="AndroidWorld Benchmark" width="70%">
</div>

*   View the full benchmark results: [Mobile AI Agents Benchmark](https://minitap.ai/research/mobile-ai-agents-benchmark)
*   Official Leaderboard: [Benchmark Leaderboard](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0)

## 🚀 Getting Started

Get up and running with Mobile-Use in a few simple steps:

1.  **Set up Environment Variables:**

    *   Copy the example `.env.example` file to `.env` and add your API keys.  An OpenAI key is required.

    ```bash
    cp .env.example .env
    ```

2.  **(Optional) Customize LLM Configuration:**

    *   To use different models or providers, create your own LLM configuration file:

    ```bash
    cp llm-config.override.template.jsonc llm-config.override.jsonc
    ```

    *   Edit `llm-config.override.jsonc` to fit your needs.

### Quick Launch (Docker - Android Only)

> [!NOTE]
> This quickstart is available for Android devices/emulators only and requires Docker.

1.  **Connect your Android device:**  Either plug in your device with USB debugging enabled, or launch an Android emulator.

> [!IMPORTANT]
> The terminal will sometimes hang while Maestro runs.  You will be asked `Maestro CLI would like to collect anonymous usage data to improve the product.` You can accept (Y) or decline (n).

2.  **Run the appropriate command for your OS:**

    *   **Linux/macOS:**

    ```bash
    chmod +x mobile-use.sh
    ./mobile-use.sh \
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" \
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

    *   **Windows (PowerShell):**

    ```powershell
    powershell.exe -ExecutionPolicy Bypass -File mobile-use.ps1 `
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" `
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

> [!NOTE]
> When using your own device, accept the ADB-related connection requests that will appear on your device. Maestro will also install its APK; accept the installation request.

#### 🧰 Troubleshooting

The script attempts to connect to your device via IP.  Ensure your device is connected to the same Wi-Fi network as your computer.

##### 1. No device IP found

If this error occurs:

```
Could not get device IP. Is a device connected via USB and on the same Wi-Fi network?
```

Determine your phone's WLAN interface via `adb shell ip addr show up`.  Then, add the `--interface <YOUR_INTERFACE_NAME>` option to the script.

##### 2. Failed to connect to <DEVICE_IP>:5555 inside Docker

This is likely a firewall issue. There isn't a single fix.

##### 3. Failed to pull GHCR docker images (unauthorized)

If you've used `ghcr.io` before for private repositories, you may have an expired token.  Try running `docker logout ghcr.io` and re-running the script.

### Manual Launch (Development Mode)

For developers, here's how to set up your environment manually:

#### 1. Device Support

Mobile-Use currently supports:

*   Physical Android Phones (USB debugging enabled)
*   Android Simulators (Android Studio setup)
*   iOS Simulators (macOS only)

> [!NOTE]
> Physical iOS devices are not yet supported.

#### 2. Prerequisites

**Android:**

*   [Android Debug Bridge (ADB)](https://developer.android.com/studio/releases/platform-tools)

**iOS:**

*   [Xcode](https://developer.apple.com/xcode/)

**General:**

*   [uv](https://github.com/astral-sh/uv) (Python package manager)
*   [Maestro](https://maestro.mobile.dev/getting-started/installing-maestro)

#### 3. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/minitap-ai/mobile-use.git && cd mobile-use
    ```

2.  [**Setup environment variables**](#-getting-started)

3.  **Create & activate the virtual environment:**

    ```bash
    # This will create a .venv directory using the Python version in .python-version
    uv venv

    # Activate the environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate
    ```

4.  **Install dependencies:**

    ```bash
    # Sync with the locked dependencies for a consistent setup
    uv sync
    ```

## 👨‍💻 Usage

To run Mobile-Use, pass your command as an argument:

**Example 1: Basic Command**

```bash
python ./src/mobile_use/main.py "Go to settings and tell me my current battery level"
```

**Example 2: Data Scraping**

```bash
python ./src/mobile_use/main.py \
  "Open Gmail, find all unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

> [!NOTE]
> If you haven't configured a specific model, Mobile-Use will prompt you to choose one from the available options.

## ❤️ Contributing

We welcome contributions!  Please see our **[Contributing Guidelines](CONTRIBUTING.md)** to get started.

## ⭐ Star History

<p align="center">
  <a href="https://star-history.com/#minitap-ai/mobile-use&Date">
    <img src="https://api.star-history.com/svg?repos=minitap-ai/mobile-use&type=Date" alt="Star History Chart" />
  </a>
</p>

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.