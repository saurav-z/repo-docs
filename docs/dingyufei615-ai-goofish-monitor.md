# AI-Powered Goofish (闲鱼) Monitor: Smartly Track & Analyze Second-Hand Goods

**Effortlessly monitor and analyze your favorite second-hand goods on Goofish (闲鱼) using AI, with a user-friendly web interface.** [See the original repo](https://github.com/dingyufei615/ai-goofish-monitor).

## Key Features:

*   ✅ **Web UI Management:** Intuitive web interface for task management, AI criteria editing, real-time log viewing, and result filtering, eliminating the need for command-line interaction.
*   🤖 **AI-Driven Task Creation:** Create complex monitoring tasks with natural language descriptions of your desired purchases.
*   ⚙️ **Concurrent Multi-Tasking:** Monitor multiple keywords simultaneously via `config.json`, with independent task execution.
*   ⚡️ **Real-time Processing:** Analyze new listings instantly, avoiding batch processing delays.
*   🧠 **Deep AI Analysis:** Leverage multi-modal large language models (e.g., GPT-4o) for in-depth analysis of product descriptions, images, and seller profiles.
*   ⚙️ **Highly Customizable:** Configure each task with unique keywords, price ranges, filtering conditions, and AI analysis prompts.
*   🔔 **Instant Notifications:** Receive immediate notifications via [ntfy.sh](https://ntfy.sh/), WeChat group bots, and [Bark](https://bark.day.app/) for items meeting your criteria.
*   📅 **Scheduled Task Execution:** Utilize Cron expressions for automated, scheduled task runs.
*   🐳 **Docker Deployment:** Simplified deployment with `docker-compose` for fast, standardized containerization.
*   🛡️ **Robust Anti-Scraping:** Mimics human behavior with random delays and user actions for increased stability.

## Key Highlights:

### Task Management
<br>
![后台任务管理](static/img.png)

### Monitoring & Notification
<br>
![后台监控截图](static/img_1.png)

### Notification on Mobile
<br>
![ntf通知截图](static/img_2.png)

## 🚀 Getting Started: (Web UI Recommended)

The Web UI provides the best user experience and is recommended.

### Step 1: Environment Setup

> ⚠️ **Python Version:** Python 3.10 or higher is recommended for local deployment and debugging. Lower versions might cause installation failures (e.g., `ModuleNotFoundError: No module named 'PIL'`).

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

    *   Windows:

        ```cmd
        copy .env.example .env
        ```

    *   Linux/MacOS:

        ```bash
        cp .env.example .env
        ```

    Environment Variable | Description | Required | Notes
    ----------------------|--------------------------------------------------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------------------------------------
    `OPENAI_API_KEY`     | Your AI model service provider's API Key.                                                                    | Yes      |  For certain local or proxy services, this may be optional.
    `OPENAI_BASE_URL`    | AI model API endpoint (OpenAI format compatible).                                                            | Yes      | Enter the API's base path, e.g., `https://ark.cn-beijing.volces.com/api/v3/`.
    `OPENAI_MODEL_NAME`  | The specific model you want to use.                                                                          | Yes      | **Must** select a multi-modal model supporting image analysis (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`).
    `PROXY_URL`          | (Optional) HTTP/S proxy for accessing the internet (if needed).                                                     | No       | Supports `http://` and `socks5://` formats (e.g., `http://127.0.0.1:7890`).
    `NTFY_TOPIC_URL`     | (Optional) [ntfy.sh](https://ntfy.sh/) topic URL for sending notifications.                                  | No       | Leave blank to disable ntfy notifications.
    `GOTIFY_URL`         | (Optional) Gotify service address.                                                                           | No       | Example: `https://push.example.de`.
    `GOTIFY_TOKEN`       | (Optional) Gotify application token.                                                                         | No       |
    `BARK_URL`           | (Optional) [Bark](https://bark.day.app/) push address.                                                        | No       | Example: `https://api.day.app/your_key`.  Leave blank to disable Bark notifications.
    `WX_BOT_URL`         | (Optional) Enterprise WeChat group bot webhook URL.                                                          | No       | Leave blank to disable WeChat notifications. Ensure URLs are enclosed in double quotes in the `.env` file.
    `WEBHOOK_URL`        | (Optional) Generic Webhook URL.                                                                            | No       | Leave blank to disable Webhook notifications.
    `WEBHOOK_METHOD`     | (Optional) Webhook request method.  Supports `GET` or `POST` (default: POST).                                | No       |
    `WEBHOOK_HEADERS`    | (Optional) Custom Webhook request headers. Must be a valid JSON string (e.g., `'{"Authorization": "Bearer xxx"}'`).| No       |
    `WEBHOOK_CONTENT_TYPE`| (Optional) POST request content type.  Supports `JSON` or `FORM` (default: JSON).                          | No       |
    `WEBHOOK_QUERY_PARAMETERS` | (Optional) GET request query parameters.  JSON string with `{{title}}` and `{{content}}` placeholders.          | No       |
    `WEBHOOK_BODY`       | (Optional) POST request body.  JSON string with `{{title}}` and `{{content}}` placeholders.                          | No       |
    `LOGIN_IS_EDGE`      |  Whether to use the Edge browser for login and scraping.                                                     | No       |  Defaults to `false`, using Chrome/Chromium.
    `PCURL_TO_MOBILE`    |  Whether to convert PC product links in notifications to mobile links.                                     | No       |  Defaults to `true`.
    `RUN_HEADLESS`       |  Run the crawler browser in headless mode.                                                                     | No       |  Defaults to `true`. Set to `false` for manual CAPTCHA handling in local debugging.  **Must be `true` for Docker deployment.**
    `AI_DEBUG_MODE`      |  Enable AI debug mode.                                                                                        | No       |  Defaults to `false`. Prints detailed AI request/response logs to the console.
    `SKIP_AI_ANALYSIS`   |  Skip AI analysis and send notifications directly.                                                           | No       |  Defaults to `false`.  Set to `true` to send notifications without AI analysis.
    `ENABLE_THINKING`    |  Enable the `enable_thinking` parameter.                                                                      | No       |  Defaults to `false`.  Some AI models require this; others don't.  If you get the error "Invalid JSON payload received. Unknown name "enable_thinking"", try setting to `false`.
    `SERVER_PORT`        |  Web UI service port.                                                                                         | No       |  Defaults to `8000`.
    `WEB_USERNAME`       | Web UI login username.                                                                                       | No       | Defaults to `admin`. **Change in production!**
    `WEB_PASSWORD`       | Web UI login password.                                                                                       | No       | Defaults to `admin123`. **Change to a strong password in production!**

    > 💡 **Debugging Tip**: If you encounter 404 errors when configuring the AI API, debug with APIs from AliCloud or Volcano, before trying other providers.  Some APIs may have compatibility issues.

    > 🔐 **Security Reminder**:  The Web UI uses Basic Authentication. The default username/password is `admin` / `admin123`. **Change these in production!**

2.  **Get Login State (Crucial!)**:  Provide valid login credentials for the crawler to access Goofish.

    **Recommended: Update via Web UI**
    1.  Skip this step and start the Web server (Step 3).
    2.  Open the Web UI and go to "System Settings."
    3.  Find "Login State File" and click "Manual Update."
    4.  Follow the instructions in the popup:
        *   Install the [Goofish Login State Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        *   Open and log in to the Goofish website.
        *   Click the extension icon and click "Extract Login State."
        *   Click "Copy to Clipboard."
        *   Paste the copied content into the Web UI and save.

    This method avoids running a GUI program on your server.

    **Alternative: Run Login Script (if you can run a GUI browser)**
    ```bash
    python login.py
    ```
    A browser window will open.  Scan the QR code with your phone's Goofish app to log in.  The script will close and generate `xianyu_state.json`.

### Step 3: Start the Web Server

```bash
python web_server.py
```

### Step 4: Start Using

1.  Open `http://127.0.0.1:8000` in your browser.
2.  Go to "Task Management" and click "Create New Task."
3.  Describe your needs in natural language (e.g., "I want to buy a Sony A7M4 camera, 95% new or better, budget under 13,000, shutter count less than 5000").
4.  Fill in task name and keywords, and click create.  The AI will generate the analysis criteria.
5.  Go back to the main interface, add a schedule or start the task.

## 🐳 Docker Deployment (Recommended)

Docker enables fast, reliable, and consistent deployments.

### Step 1: Environment Prep (Similar to Local)

1.  **Install Docker**:  Ensure Docker Engine is installed ([Docker Engine Install](https://docs.docker.com/engine/install/)).

2.  **Clone and Configure**:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env`**:  Create and populate the `.env` file as described in the [Getting Started](#-getting-started-web-ui-recommended) section.

4.  **Get Login State (Critical!)**:  You *must* set the login state *after* starting the container, via the Web UI:
    1.  Run `docker-compose up -d` on the host.
    2.  Open `http://127.0.0.1:8000` in your browser.
    3.  Go to "System Settings" and click "Manual Update."
    4.  Follow the popup instructions to extract and paste your login state.

> ℹ️ **Python Version**:  Docker uses Python 3.11 from the Dockerfile, so Python compatibility is not a local concern.

### Step 2: Run the Docker Container

The project includes `docker-compose.yaml`.

Run the following in the project root:

```bash
docker-compose up --build -d
```

This starts the service in the background.  `docker-compose` uses the `.env` and `docker-compose.yaml` configurations.

If you encounter network issues in the container, troubleshoot or use a proxy.

> ⚠️ **OpenWrt Deployment Notes**: OpenWrt routers may experience DNS resolution problems with the default Docker Compose network. If you get `ERR_CONNECTION_REFUSED`, check your container network configuration and consider manually configuring DNS or adjusting the network mode.

### Step 3: Access and Manage

-   **Access Web UI**: Open `http://127.0.0.1:8000` in your browser.
-   **View Real-time Logs**:  `docker-compose logs -f`
-   **Stop Container**:  `docker-compose stop`
-   **Start Stopped Container**:  `docker-compose start`
-   **Stop and Remove Container**: `docker-compose down`

## 📸 Web UI Functionality

-   **Task Management**:
    -   AI-powered task creation: Use natural language for monitoring task generation and AI analysis standard creation.
    -   Visual editing and control: Modify task parameters (keywords, prices, scheduling, etc.) in a table and independently start/stop/delete tasks.
    -   Scheduled tasks: Configure Cron expressions for automated, recurring task runs.
-   **Result Viewing**:
    -   Card-style browsing: Clearly displays compliant product details.
    -   Smart filtering and sorting: Filters AI-recommended products and sorts by crawl time, publication time, price, and more.
    -   Deep details: Click to view detailed data and AI analysis results (JSON).
-   **Run Logs**:
    -   Real-time log streams: View detailed crawler logs in real time, facilitating progress tracking and troubleshooting.
    -   Log management: Supports auto-refresh, manual refresh, and one-click log clearing.
-   **System Settings**:
    -   Status check: One-click check of key dependencies, such as `.env` settings and login status.
    -   Prompt editing: Edit and save prompt files directly in the web interface, for real-time adjustment of AI thinking logic.

## 🚀 Workflow

```mermaid
graph TD
    A[Start Monitoring Task] --> B[Task: Search Items];
    B --> C{New Item Found?};
    C -- Yes --> D[Fetch Item Details & Seller Info];
    D --> E[Download Item Images];
    E --> F[Call AI for Analysis];
    F --> G{AI Recommends?};
    G -- Yes --> H[Send Notification];
    H --> I[Save Record to JSONL];
    G -- No --> I;
    C -- No --> J[Next Page/Wait];
    J --> B;
    I --> C;
```

## 🔐 Web Interface Authentication

### Authentication Configuration

The Web interface uses Basic Authentication to protect the management interface and APIs.

#### Configuration

Set credentials in `.env`:

```bash
# Web Service Authentication Settings
WEB_USERNAME=admin
WEB_PASSWORD=admin123
```

#### Default Credentials

If not set in `.env`, the defaults are:

-   Username: `admin`
-   Password: `admin123`

**⚠️ IMPORTANT: Change these in production!**

#### Authentication Scope

-   **Protected:** All API endpoints, the Web UI, and static resources.
-   **Unprotected:** Health check endpoint (`/health`).

#### Usage

1.  **Browser Access:** The browser will prompt for credentials.
2.  **API Calls:** Include Basic Authentication headers in the requests.
3.  **Frontend JavaScript:** Handles authentication automatically.

#### Security Recommendations

1.  Change the default password to a strong password.
2.  Use HTTPS in production.
3.  Change credentials periodically.
4.  Restrict IP access via a firewall.

See [AUTH_README.md](AUTH_README.md) for details.

## Frequently Asked Questions (FAQ)

Find detailed answers to common questions about setup, AI configuration, and anti-scraping strategies.

👉 **[Click here to see the FAQ (FAQ.md)](FAQ.md)**

## Acknowledgments

Thanks to the following projects:

-   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

And thanks to the LinuxDo community and contributors.

-   [@jooooody](https://linux.do/u/jooooody/summary)

And thank you to ClaudeCode/ModelScope/Gemini for the models/tools and liberating from tedious Vibe Coding.

## Disclaimer

-   Please abide by the Goofish terms of service and robots.txt rules. Avoid excessively frequent requests.
-   This project is for educational and research purposes only.
-   It is released under the [MIT License](LICENSE).
-   The project author and contributors are not responsible for any damages or losses resulting from using this software.
-   See [DISCLAIMER.md](DISCLAIMER.md) for more information.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)