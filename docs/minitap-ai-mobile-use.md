# Mobile-Use: Automate Your Phone with Natural Language 🤖

Effortlessly control your Android or iOS device using simple, natural language commands. ✨

<div align="center">
  <img src="./doc/linkedin-demo-with-text.gif" alt="Mobile-Use in Action" />
</div>

<div align="center">
  <a href="https://discord.gg/6nSqmQ9pQs"><img src="https://img.shields.io/discord/1403058278342201394?color=7289DA&label=Discord&logo=discord&logoColor=white&style=for-the-badge" alt="Discord"></a>
  <a href="https://github.com/minitap-ai/mobile-use/stargazers"><img src="https://img.shields.io/github/stars/minitap-ai/mobile-use?style=for-the-badge&color=e0a8dd" alt="GitHub stars"></a>
  <p align="center">
    <a href="https://discord.gg/6nSqmQ9pQs"><b>Join our Discord</b></a> •
    <a href="https://x.com/minitap_ai?t=iRWtI497UhRGLeCKYQekig&s=09"><b>Follow us on X (Twitter)</b></a>
  </p>
</div>

Mobile-use is an open-source AI agent designed to automate your mobile device using natural language.  This project empowers you to control your Android or iOS device by simply speaking or typing your desired actions. From sending messages to navigating complex app interfaces, mobile-use simplifies mobile interaction. Check out the original repo [here](https://github.com/minitap-ai/mobile-use).

> Your feedback is invaluable! Join the conversation on [Discord](https://discord.gg/6nSqmQ9pQs) to suggest features, report bugs, and help shape the future of mobile-use. ❤️

## Key Features

*   🗣️ **Natural Language Control:**  Command your phone using everyday language.
*   📱 **UI-Aware Automation:**  Intelligently navigates and interacts with app interfaces.
*   📊 **Data Scraping:** Extract and format information from any app (e.g., convert to JSON) with natural language instructions.
*   🔧 **Extensible & Customizable:**  Easily configure different Large Language Models (LLMs) to power your agents.

## Benchmarks & Performance

<p align="center">
  <img src="./doc/benchmark.png" alt="Project banner" />
</p>

Mobile-use achieved #1 Open-source Pass@1 on the AndroidWorld benchmark.

*   Learn more about the benchmarks: [Mobile AI Agents Benchmark](https://minitap.ai/research/mobile-ai-agents-benchmark)
*   View the leaderboard: [Benchmark Leaderboard](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0)

## Getting Started

Follow these steps to quickly set up and use mobile-use:

### 1. Environment Setup

*   **Configure Environment Variables:**
    *   Copy the example file: `cp .env.example .env`
    *   Add your required API keys to the `.env` file.  An OpenAI key is required.

### 2. (Optional) Customize LLM Configuration

*   To use different models or providers, create your own LLM configuration file.
    *   Copy the template: `cp llm-config.override.template.jsonc llm-config.override.jsonc`
    *   Edit `llm-config.override.jsonc` to match your needs.

### 3. Quick Launch (Docker - Android Only)

> [!NOTE]
> This quickstart is only available for Android devices/emulators at this time. You must have Docker installed.

First:

*   Connect your Android device and enable USB debugging through Developer Options.
*   Or launch an Android emulator.

> [!IMPORTANT]
> At some point, the terminal will HANG, and Maestro will ask you `Maestro CLI would like to collect anonymous usage data to improve the product.`
> It's up to you whether you accept (i.e enter 'Y') or not (i.e. enter 'n').

Run these commands in your terminal:

1.  **Linux/macOS:**

    ```bash
    chmod +x mobile-use.sh
    ./mobile-use.sh \
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" \
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

2.  **Windows (PowerShell):**

    ```powershell
    powershell.exe -ExecutionPolicy Bypass -File mobile-use.ps1 `
      "Open Gmail, find first 3 unread emails, and list their sender and subject line" `
      --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
    ```

> [!NOTE]
> If you are using your own device, make sure to accept the ADB-related connection requests that will pop up on your device.
> Similarly, Maestro will need to install its APK on your device, which will also require you to accept the installation request.

#### Troubleshooting Quick Launch

*   **No Device IP Found:** Ensure your device is connected to the same Wi-Fi network as your computer.  If the script can't detect the IP, use the `--interface <YOUR_INTERFACE_NAME>` option with your device's WLAN interface.
*   **Failed to Connect to <DEVICE_IP>:5555 inside Docker:**  Check your firewall settings to ensure that the connection isn't being blocked.
*   **Failed to Pull GHCR Docker Images (Unauthorized):** If you've previously used `ghcr.io` for private repositories, you might have an expired token. Try running `docker logout ghcr.io` and retry the script.

### 4. Manual Launch (Development Mode)

#### 1. Device Support

Mobile-use currently supports:

*   **Physical Android Phones:**  Connect via USB with USB debugging enabled.
*   **Android Simulators:**  Set up through Android Studio.
*   **iOS Simulators:** Supported for macOS users.

> [!NOTE]
> Physical iOS devices are not yet supported.

#### 2. Prerequisites

For Android:

*   **[Android Debug Bridge (ADB)](https://developer.android.com/studio/releases/platform-tools)**: Required for connecting to your device.

For iOS:

*   **[Xcode](https://developer.apple.com/xcode/)**: Apple's IDE.

You'll also need:

*   **[uv](https://github.com/astral-sh/uv)**: A fast Python package manager.
*   **[Maestro](https://maestro.mobile.dev/getting-started/installing-maestro)**:  The framework used to interact with your device.

#### 3. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/minitap-ai/mobile-use.git && cd mobile-use
    ```

2.  [**Setup environment variables**](#-getting-started)

3.  **Create and activate the virtual environment:**

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

## Usage

Execute commands using `python ./src/mobile_use/main.py "your command here"`.

**Example 1: Basic Command**

```bash
python ./src/mobile_use/main.py "Go to settings and tell me my current battery level"
```

**Example 2: Data Scraping**

Extract specific information and receive it in a structured format:

```bash
python ./src/mobile_use/main.py \
  "Open Gmail, find all unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

> [!NOTE]
> If a specific model isn't configured, mobile-use will prompt you to select one from the available options.

## Contributing

We welcome your contributions!  Review our **[Contributing Guidelines](CONTRIBUTING.md)** to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.