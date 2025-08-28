# AI-Powered Goofish (Xianyu) Monitor: Smartly Track & Analyze Used Goods 🤖

**Stay ahead of the game with AI-driven real-time monitoring and intelligent analysis for Xianyu (Goofish) used goods, complete with a user-friendly web interface.** 

[View Original Repo](https://github.com/dingyufei615/ai-goofish-monitor)

## ✨ Key Features

*   ✅ **Intuitive Web UI**: Manage tasks, edit AI criteria, view logs, and filter results – all without touching the command line.
*   🧠 **AI-Driven Task Creation**: Describe your desired item in natural language, and let AI build the perfect monitoring task.
*   🔄 **Concurrent Multi-Tasking**: Monitor multiple keywords simultaneously with independent, non-interfering tasks via `config.json`.
*   ⚡️ **Real-Time Processing**: Analyze new listings instantly, eliminating batch processing delays.
*   💡 **Deep AI Analysis**: Leverage multimodal LLMs (like GPT-4o) to analyze item descriptions, images, and seller profiles for precise filtering.
*   ⚙️ **Highly Customizable**: Tailor each task with unique keywords, price ranges, filters, and AI analysis prompts.
*   🔔 **Instant Notifications**: Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat group bots, and [Bark](https://bark.day.app/) for immediate updates.
*   📅 **Scheduled Task Execution**: Utilize Cron expressions to schedule tasks for automated monitoring.
*   🐳 **Docker Ready**: Deploy quickly with the provided `docker-compose` configuration.
*   🛡️ **Robust Anti-Scraping**: Mimics human behavior with random delays and user actions for enhanced stability.

## 🖼️ Screenshots

**(See the original README for images of the web UI and notifications)**

## 🚀 Getting Started (Web UI Recommended)

The Web UI offers the best user experience.

### Step 1: Environment Setup

> ⚠️ **Python Version**: Use Python 3.10 or higher for local deployment and debugging. Older versions may cause dependency installation failures (e.g., `ModuleNotFoundError: No module named 'PIL'`).

1.  Clone the project:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Configuration

1.  **Configure Environment Variables**: Copy `.env.example` to `.env` and modify the values.

    ```bash
    # Windows
    copy .env.example .env
    # Linux/MacOS
    cp .env.example .env
    ```

    Key environment variables:

    | Variable            | Description                                   | Required? | Notes                                                                         |
    | :------------------ | :-------------------------------------------- | :-------- | :---------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`    | Your AI model provider's API key.             | Yes       | May be optional for certain local or proxy services.                          |
    | `OPENAI_BASE_URL`   | API endpoint for your AI model.              | Yes       | Must be compatible with OpenAI format, e.g., `https://ark.cn-beijing.volces.com/api/v3/`. |
    | `OPENAI_MODEL_NAME` | The specific model name to use.            | Yes       | **REQUIRED**: Choose a multimodal model supporting image analysis, e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`. |
    | `PROXY_URL`         | (Optional) HTTP/S proxy for bypassing restrictions. | No        | Supports `http://` and `socks5://`, e.g., `http://127.0.0.1:7890`.             |
    | ... (See original README for full table) ...

    > 💡 **Debugging Tip**: If you get 404 errors with AI APIs, test with the APIs provided by Ali or Volcano for preliminary testing and ensure basic features are fine before proceeding. Some APIs might have compatibility issues or need special settings.

    > 🔐 **Security Note**: The Web UI uses Basic Authentication. Change the default username/password (`admin`/`admin123`) in production!

2.  **Get Login Credentials (Essential!)**: Acquire valid login credentials for Xianyu. The Web UI is the recommended method:

    **Recommended: Via Web UI**
    1.  Skip this and start the Web service in step 3.
    2.  Navigate to "System Settings" in the Web UI.
    3.  Find "Login State File" and click "Manual Update".
    4.  Follow the instructions in the popup:
        -   Install the [Xianyu Login State Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        -   Log in to Xianyu.
        -   Click the extension icon and click "Extract Login State".
        -   Click "Copy to Clipboard".
        -   Paste into the Web UI.

    **Alternative: Login Script** (for local/desktop environments)

    ```bash
    python login.py
    ```

    This will open a browser. Scan the QR code with your Xianyu app. It will generate a `xianyu_state.json` file.

### Step 3: Start the Web Server

```bash
python web_server.py
```

### Step 4: Start Monitoring

1.  Open `http://127.0.0.1:8000` in your browser.
2.  In "Task Management", click "Create New Task".
3.  Describe your needs (e.g., "Looking for a Sony A7M4 camera, 95% new, budget under 13000 yuan, shutter count below 5000"), and fill in task details.
4.  Click "Create", and the AI generates analysis criteria.
5.  Add a schedule or start the task manually.

## 🐳 Docker Deployment (Recommended)

Docker simplifies deployment.

### Step 1: Environment Preparation

1.  **Install Docker:** Ensure you have [Docker Engine](https://docs.docker.com/engine/install/) installed.
2.  **Clone and Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env` File**: Create and populate `.env` as described in "Getting Started".
4.  **Get Login Credentials (CRITICAL!)**: As the docker container does not allow for easy QR code scanning, the recommended option is to extract the login state using the Web UI:

    1.  (On your host machine) Run `docker-compose up -d`.
    2.  Open `http://127.0.0.1:8000` in your browser.
    3.  Go to "System Settings" and click "Manual Update".
    4.  Follow the instructions in the popup:
        -   Install the [Xianyu Login State Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        -   Log in to Xianyu.
        -   Click the extension icon and click "Extract Login State".
        -   Click "Copy to Clipboard".
        -   Paste into the Web UI.

> ℹ️ **Python Version:** Docker uses Python 3.11, as specified in the Dockerfile.

### Step 2: Run Docker Container

Use `docker-compose.yaml`.

```bash
docker-compose up --build -d
```

### Step 3: Access and Manage

-   **Web UI**: `http://127.0.0.1:8000`
-   **Logs**: `docker-compose logs -f`
-   **Stop**: `docker-compose stop`
-   **Start**: `docker-compose start`
-   **Down**: `docker-compose down`

## 🛠️ Web UI Feature Overview

**(See original README for a detailed list of Web UI features)**

## ⚙️ Workflow

**(See original README for the workflow diagram)**

## 🔐 Web Interface Authentication

**(See original README for detailed authentication configuration)**

## ❓ FAQ

👉 **[See the FAQ (FAQ.md)](FAQ.md)**

## 🙏 Acknowledgements

**(See original README for a list of projects and communities this is based on)**

## ⚠️ Important Notes

**(See original README for important notes regarding usage, licenses, and disclaimers)**

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)