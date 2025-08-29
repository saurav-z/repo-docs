# AI-Powered Goofish (Xianyu) Monitor: Stay Ahead of the Deals

**Get real-time notifications and AI-powered analysis for your desired items on Xianyu (Goofish) with this powerful and user-friendly monitoring tool!**

[View the original repository on GitHub](https://github.com/dingyufei615/ai-goofish-monitor)

## ✨ Key Features

*   ✅ **Intuitive Web UI:** Manage tasks, edit AI criteria, view logs, and filter results directly from a web interface – no command-line fuss!
*   💬 **AI-Driven Task Creation:** Describe your ideal purchase in natural language, and the AI will generate a complex monitoring task with intelligent filtering.
*   🚀 **Concurrent Multi-Tasking:** Monitor multiple keywords simultaneously via `config.json`, each running independently.
*   ⏱️ **Real-Time Streaming:** Get instant analysis and notifications as new items appear.
*   🧠 **Deep AI Analysis:** Integrate multi-modal large language models (e.g., GPT-4o) to analyze item descriptions, images, and seller profiles for precise filtering.
*   ⚙️ **Highly Customizable:** Tailor each task with independent keywords, price ranges, filters, and AI analysis prompts.
*   🔔 **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat Work group bots, and [Bark](https://bark.day.app/) to your phone or desktop.
*   📅 **Scheduled Tasks:** Utilize Cron expressions for flexible, automated task scheduling.
*   🐳 **Docker for Easy Deployment:** Deploy quickly and consistently with a pre-configured `docker-compose` setup.
*   🛡️ **Robust Anti-Scraping Measures:** Benefit from realistic user behavior, including random delays, to enhance stability and bypass anti-bot mechanisms.

## 🖼️ Screenshots

**Web UI - Task Management**
![img.png](static/img.png)

**Web UI - Monitoring Dashboard**
![img_1.png](static/img_1.png)

**Notification Example (ntfy.sh)**
![img_2.png](static/img_2.png)

## 🚀 Quick Start (Web UI Recommended)

The Web UI offers the best user experience.

### Step 1: Environment Setup

> ⚠️ **Python Requirement:** Python 3.10+ is highly recommended for local deployment and debugging. Lower versions may cause installation failures or runtime errors (e.g., `ModuleNotFoundError: No module named 'PIL'`).

1.  Clone the repository:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

2.  Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Configuration

1.  **Configure Environment Variables:** Copy `.env.example` to `.env` and modify the values.

    Windows:

    ```cmd
    copy .env.example .env
    ```

    Linux/MacOS:

    ```bash
    cp .env.example .env
    ```

    Environment Variables:

    | Variable          | Description                                                        | Required | Notes                                                                                                                                                                                                                         |
    | :---------------- | :----------------------------------------------------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
    | `OPENAI_API_KEY`  | Your AI model provider's API Key.                                | Yes      | May be optional for local or specific proxy services.                                                                                                                                                                     |
    | `OPENAI_BASE_URL` | The API endpoint for your AI model, compatible with OpenAI format. | Yes      | Specify the base URL, e.g., `https://ark.cn-beijing.volces.com/api/v3/`.                                                                                                                                                 |
    | `OPENAI_MODEL_NAME` | The specific model name you wish to use. | Yes      | **Must** select a multi-modal model that supports image analysis, such as `doubao-seed-1-6-250615`, `gemini-2.5-pro`, etc. |
    | `PROXY_URL`       | (Optional) HTTP/S proxy for bypassing network restrictions.          | No       | Supports `http://` and `socks5://` formats. E.g., `http://127.0.0.1:7890`.                                                                                                                                      |
    | `NTFY_TOPIC_URL`  | (Optional) [ntfy.sh](https://ntfy.sh/) topic URL for notifications.  | No       | If left empty, ntfy notifications will be disabled.                                                                                                                                                                  |
    | `GOTIFY_URL`      | (Optional) Gotify service address.                                 | No       | e.g., `https://push.example.de`.                                                                                                                                                                                    |
    | `GOTIFY_TOKEN`    | (Optional) Gotify application token.                               | No       |                                                                                                                                                                                                                           |
    | `BARK_URL`        | (Optional) [Bark](https://bark.day.app/) push address.             | No       | e.g., `https://api.day.app/your_key`. If left blank, Bark notifications will be disabled.                                                                                                                            |
    | `WX_BOT_URL`      | (Optional) WeChat Work group bot Webhook address.                   | No       | If left blank, WeChat Work notifications will be disabled.                                                                                                                                                           |
    | `WEBHOOK_URL`     | (Optional) Generic Webhook URL.                                      | No       | If left blank, generic Webhook notifications will be disabled.                                                                                                                                                           |
    | `WEBHOOK_METHOD`  | (Optional) Webhook request method.                                 | No       | Supports `GET` or `POST`, defaults to `POST`.                                                                                                                                                                       |
    | `WEBHOOK_HEADERS` | (Optional) Custom Webhook request headers.                          | No       | Must be a valid JSON string, e.g., `'{"Authorization": "Bearer xxx"}'`.                                                                                                                                           |
    | `WEBHOOK_CONTENT_TYPE` | (Optional) POST request content type.                          | No       | Supports `JSON` or `FORM`, defaults to `JSON`.                                                                                                                                                                     |
    | `WEBHOOK_QUERY_PARAMETERS` | (Optional) GET request query parameters.                          | No       | JSON string, supports `{{title}}` and `{{content}}` placeholders.                                                                                                                                   |
    | `WEBHOOK_BODY` | (Optional) POST request body.                          | No       | JSON string, supports `{{title}}` and `{{content}}` placeholders.                                                                                                                                   |
    | `LOGIN_IS_EDGE`   | Use Edge browser for login and scraping.                             | No       | Defaults to `false` (Chrome/Chromium).                                                                                                                                                                     |
    | `PCURL_TO_MOBILE`   | Convert PC product links to mobile version in the notification.                             | No       | Defaults to `true`.                                                                                                                                                                     |
    | `RUN_HEADLESS`    | Run the crawler browser in headless mode.                           | No       | Defaults to `true`. Set to `false` for local debugging if you encounter captcha. **Must be `true` for Docker deployment.**                                                                             |
    | `AI_DEBUG_MODE`   | Enable AI debug mode.                                              | No       | Defaults to `false`.  Prints detailed AI request/response logs to the console.                                                                                                                                 |
    | `SKIP_AI_ANALYSIS` | Skip AI analysis and send notifications directly.                | No       | Defaults to `false`.  If set to `true`, all scraped items will be notified without AI analysis.                                                                                                                   |
    | `ENABLE_THINKING` | Enable the `enable_thinking` parameter.                | No       | Defaults to `false`.  Some AI models require this parameter, while some do not support it. If you encounter the error "Invalid JSON payload received. Unknown name "enable_thinking"", try setting it to `false`.                                                                                                                   |
    | `SERVER_PORT`     | Web UI service port.                                               | No       | Defaults to `8000`.                                                                                                                                                                                        |
    | `WEB_USERNAME`    | Web UI login username.                                               | No       | Defaults to `admin`. **Important: Change in production.**                                                                                                                                                  |
    | `WEB_PASSWORD`    | Web UI login password.                                               | No       | Defaults to `admin123`. **Important: Change to a strong password in production!**                                                                                                                            |

    > 💡 **Debugging Tip:** If you encounter 404 errors when configuring your AI API, try using APIs from Alibaba Cloud or Volcano Engine first to ensure basic functionality before testing other providers. Some providers may have compatibility issues or require specific configurations.

    > 🔐 **Security Reminder:** Basic Authentication is enabled for the Web UI. The default username and password are `admin` / `admin123`. **Change these to strong credentials in production!**

2.  **Obtain Login State (Critical!)**: You need valid login credentials for the crawler. The Web UI is the easiest way:

    **Recommended: Update via Web UI**

    1.  Skip this step and proceed to step 3 to start the Web service.
    2.  After opening the Web UI, navigate to the **"System Settings"** page.
    3.  Find the "Login Status File" section, and click the **"Manual Update"** button.
    4.  Follow the instructions in the popup:
        -   Install the [Xianyu Login State Extraction Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        -   Open Xianyu's official website in Chrome and log in.
        -   After successful login, click the extension icon in the Chrome toolbar.
        -   Click the "Extract Login State" button.
        -   Click the "Copy to Clipboard" button.
        -   Paste the copied content into the Web UI and save it.

    This method is the most convenient, as it does not require running a program with a graphical interface on the server.

    **Alternative: Run Login Script**

    If you can run a program locally or on a server with a desktop environment, you can use the traditional script method:

    ```bash
    python login.py
    ```

    This will open a browser window. Please use the **Xianyu App on your phone to scan the QR code** to log in. Upon successful login, the program will close automatically and generate an `xianyu_state.json` file in the project root directory.

### Step 3: Start the Web Server

Once ready, launch the Web UI server:

```bash
python web_server.py
```

### Step 4: Get Started

Open `http://127.0.0.1:8000` in your browser to access the Web UI.

1.  In the **"Task Management"** page, click **"Create New Task"**.
2.  In the window that appears, describe your desired purchase in natural language (e.g., "I want to buy a Sony A7M4 camera, mint condition or better, budget under 13,000, less than 5000 shutter count"), and fill in the task name and keywords.
3.  Click "Create," and the AI will automatically generate a sophisticated set of analysis criteria.
4.  Go back to the main page, schedule the task or click "Start" to begin automated monitoring!

## 🐳 Docker Deployment (Recommended)

Docker simplifies deployment by packaging the application and its dependencies into a standardized unit.

### Step 1: Environment Preparation (Similar to Local Setup)

1.  **Install Docker**: Ensure Docker Engine is installed on your system (see [Docker Engine Installation](https://docs.docker.com/engine/install/)).

2.  **Clone and Configure the Project**:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create the `.env` File**: Create a `.env` file in the project root and fill it with the appropriate values. See the **[Quick Start](#-quick-start-web-ui-recommended)** section for details.

4.  **Obtain Login State (Essential!)**:  Docker containers cannot perform QR code login.  **After** starting the container, you'll need to set the login state via the Web UI:
    1. (On the host machine) Execute `docker-compose up -d` to start the service.
    2.  Open `http://127.0.0.1:8000` in your browser to access the Web UI.
    3.  Navigate to the **"System Settings"** page, and click the **"Manual Update"** button.
    4.  Follow the instructions in the popup:
        -   Install the [Xianyu Login State Extraction Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        -   Open Xianyu's official website in Chrome and log in.
        -   After successful login, click the extension icon in the Chrome toolbar.
        -   Click the "Extract Login State" button.
        -   Click the "Copy to Clipboard" button.
        -   Paste the copied content into the Web UI and save it.

> ℹ️ **Regarding Python Version:** The project uses Python 3.11, specified in the Dockerfile. There are no compatibility issues with your local Python version.

### Step 2: Run the Docker Container

The project includes a `docker-compose.yaml` file. Use `docker-compose` for container management, as it is easier than `docker run`.

In the project root, run:

```bash
docker-compose up --build -d
```

This starts the service in detached mode. `docker-compose` automatically reads the `.env` and `docker-compose.yaml` files to build and start the container.

If you encounter network problems inside the container, troubleshoot the network settings or use a proxy.

> ⚠️ **OpenWrt Deployment Notes:** If deploying this application on an OpenWrt router, you might encounter DNS resolution issues. This is due to the default network created by Docker Compose failing to inherit the OpenWrt DNS settings correctly. If you encounter `ERR_CONNECTION_REFUSED` errors, examine your container's network configuration, and consider manually configuring DNS or adjusting the network mode to ensure the container can access the external network.

### Step 3: Access and Management

-   **Access Web UI:** Open `http://127.0.0.1:8000` in your browser.
-   **View Real-time Logs:** `docker-compose logs -f`
-   **Stop the Container:** `docker-compose stop`
-   **Start a Stopped Container:** `docker-compose start`
-   **Stop and Remove the Container:** `docker-compose down`

## 📸 Web UI Feature Overview

-   **Task Management:**
    -   **AI Task Creation:** Describe your needs in natural language to create monitoring tasks and AI analysis criteria with a single click.
    -   **Visual Editing and Control:** Directly modify task parameters (keywords, price, scheduling rules, etc.) in a table and independently start/stop or delete each task.
    -   **Scheduled Tasks:** Configure Cron expressions for automated, periodic task execution.
-   **Result Viewing:**
    -   **Card View:** Display eligible items clearly as image-and-text cards.
    -   **Smart Filtering and Sorting:** Filter items marked as "recommended" by the AI with a single click, and sort by crawl time, publish time, price, and more.
    -   **Deep Details:** Click to view each item's complete data and AI analysis results in JSON format.
-   **Running Logs:**
    -   **Real-Time Log Stream:** Monitor the detailed logs of the crawler's operations in real-time on the webpage, facilitating progress tracking and troubleshooting.
    -   **Log Management:** Support automatic refresh, manual refresh, and one-click clearing of logs.
-   **System Settings:**
    -   **Status Checks:** Check key dependencies like `.env` configurations and login status with one click.
    -   **Prompt Editing:** Edit and save the `prompt` file used for AI analysis directly on the webpage, and adjust the AI's reasoning logic in real-time.

## 🚀 Workflow Diagram

This diagram illustrates the core processing logic of a single monitoring task from initiation to completion. The `web_server.py` acts as the main service and launches one or more of these task processes based on user interactions or scheduled tasks.

```mermaid
graph TD
    A[Start Monitoring Task] --> B[Task: Search Items];
    B --> C{New Item Found?};
    C -- Yes --> D[Fetch Item Details & Seller Info];
    D --> E[Download Item Images];
    E --> F[Call AI for Analysis];
    F --> G{AI Recommended?};
    G -- Yes --> H[Send Notification];
    H --> I[Save to JSONL];
    G -- No --> I;
    C -- No --> J[Next Page/Wait];
    J --> B;
    I --> C;
```

## 🔐 Web UI Authentication

### Authentication Configuration

The Web UI uses Basic Authentication to ensure only authorized users can access the management interface and API.

#### Configuration

Set authentication credentials in the `.env` file:

```bash
# Web service authentication settings
WEB_USERNAME=admin
WEB_PASSWORD=admin123
```

#### Default Credentials

If you don't set authentication credentials in `.env`, the system will use these defaults:

-   Username: `admin`
-   Password: `admin123`

**⚠️ Important: Change the default password in production!**

#### Authentication Scope

-   **Requires Authentication:** All API endpoints, Web UI, static resources
-   **No Authentication Required:** Health check endpoint (`/health`)

#### Usage

1.  **Browser Access:** The authentication dialog appears when accessing the Web UI.
2.  **API Calls:** You must include Basic Authentication information in the request headers.
3.  **Frontend JavaScript:** The frontend handles authentication automatically; no modifications are needed.

#### Security Recommendations

1.  Change the default password to a strong password.
2.  Use HTTPS in production.
3.  Change the authentication credentials periodically.
4.  Restrict access by IP address via a firewall.

For more information, see [AUTH_README.md](AUTH_README.md).

## Frequently Asked Questions (FAQ)

A detailed FAQ document is available to address common questions, from environment setup and AI settings to anti-scraping strategies.

👉 **[Click here to view the FAQ (FAQ.md)](FAQ.md)**

## Acknowledgements

This project incorporates inspiration and code from these excellent projects:

-   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

And thanks to the contributions of the LinuxDo community:

-   [@jooooody](https://linux.do/u/jooooody/summary)

And thanks to the support from ClaudeCode/ModelScope/Gemini for releasing models and tools, which makes "Vibe Coding" become true and fun.

## ⚠️ Important Notes

-   Please adhere to Xianyu's user agreements and robots.txt rules. Avoid excessive requests to prevent server overload or account restrictions.
-   This project is for educational and research purposes only. Do not use it for illegal activities.
-   This project is released under the [MIT License](LICENSE), provided "as is," without any warranties.
-   The project author and contributors are not liable for any direct, indirect, incidental, or special damages resulting from the use of this software.
-   Please refer to the [Disclaimer](DISCLAIMER.md) for more detailed information.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)