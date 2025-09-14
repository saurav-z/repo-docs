# AI-Powered Goofish (Xianyu) Monitor: Effortlessly Track and Analyze Second-Hand Goods

[View the original repository on GitHub](https://github.com/dingyufei615/ai-goofish-monitor)

This project provides an AI-driven monitoring and analysis tool for Xianyu (Goofish), offering real-time tracking, smart filtering, and a user-friendly web interface for managing your secondhand item hunts.

## Key Features:

*   **Intuitive Web UI:** Manage tasks, edit AI criteria, view real-time logs, and filter results through a complete web interface.
*   **AI-Driven Task Creation:** Describe your desired item in natural language, and let the AI generate a monitoring task with complex filtering.
*   **Concurrent Multi-Tasking:** Monitor multiple keywords simultaneously with independent, non-interfering tasks via `config.json`.
*   **Real-time Streaming:** Analyze new items immediately upon discovery, eliminating batch processing delays.
*   **Deep AI Analysis:** Integrate multimodal large language models (e.g., GPT-4o) to analyze item descriptions, images, and seller profiles for precise filtering.
*   **Highly Customizable:** Configure individual keywords, price ranges, filters, and AI analysis prompts (instructions) for each monitoring task.
*   **Instant Notifications:** Receive notifications via [ntfy.sh](https://ntfy.sh/), WeChat Work group bots, and [Bark](https://bark.day.app/) for AI-recommended items.
*   **Scheduled Task Execution:** Utilize Cron expressions for automated, periodic task scheduling.
*   **Dockerized Deployment:** Deploy quickly and consistently with a pre-configured `docker-compose` setup.
*   **Robust Anti-Scraping:** Mimic human behavior with random delays and user actions to enhance stability and avoid detection.

## Core Functionality

**Work Flow:**

```mermaid
graph TD
    A[Start Monitor Task] --> B[Task: Search Items];
    B --> C{New Item Found?};
    C -- Yes --> D[Fetch Item Details & Seller Info];
    D --> E[Download Item Images];
    E --> F[Call AI for Analysis];
    F --> G{AI Recommends?};
    G -- Yes --> H[Send Notification];
    H --> I[Save to JSONL];
    G -- No --> I;
    C -- No --> J[Next Page/Wait];
    J --> B;
    I --> C;
```

## Getting Started

**Recommended: Use the Web UI for the best experience.**

### Step 1: Environment Setup

> ⚠️ **Python Version:** Use Python 3.10 or higher for local development and debugging to avoid dependency installation errors (e.g., `ModuleNotFoundError: No module named 'PIL'`).

1.  Clone the repository:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Configuration

1.  **Configure Environment Variables:**  Copy `.env.example` to `.env` and modify its contents.

    *   **Windows:**

        ```cmd
        copy .env.example .env
        ```

    *   **Linux/macOS:**

        ```shell
        cp .env.example .env
        ```

    Here's a table of available environment variables:

    | Variable          | Description                                           | Required? | Notes                                                                                                                                                 |
    | ----------------- | ----------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`  | Your AI model provider's API Key.                    | Yes       |                                                                                                                                                       |
    | `OPENAI_BASE_URL` | Base URL for the AI model API.                        | Yes       | Must be compatible with OpenAI format.  e.g., `https://ark.cn-beijing.volces.com/api/v3/`.                                                         |
    | `OPENAI_MODEL_NAME` | The specific model you want to use.                 | Yes       | **Crucial:** Choose a multimodal model supporting image analysis, such as `doubao-seed-1-6-250615`, `gemini-2.5-pro` etc.                                  |
    | `PROXY_URL`       | (Optional) HTTP/S proxy for bypassing geo-restrictions. | No        | Supports `http://` and `socks5://` formats, e.g., `http://127.0.0.1:7890`.                                                                          |
    | `NTFY_TOPIC_URL`  | (Optional) [ntfy.sh](https://ntfy.sh/) topic URL for notifications.          | No        | Leave empty to disable ntfy notifications.                                                                                                           |
    | `GOTIFY_URL`      | (Optional) Gotify service address.                                          | No        | e.g., `https://push.example.de`.                                                                                                                 |
    | `GOTIFY_TOKEN`    | (Optional) Gotify application token.                                            | No        |                                                                                                                                                       |
    | `BARK_URL`        | (Optional) [Bark](https://bark.day.app/) push address.                           | No        | e.g., `https://api.day.app/your_key`. Leave empty to disable Bark notifications.                                                                     |
    | `WX_BOT_URL`      | (Optional) WeChat Work group bot webhook URL.                              | No        | Leave empty to disable WeChat Work notifications. **Note:** Enclose the URL in double quotes in `.env` to prevent configuration issues.              |
    | `WEBHOOK_URL`      | (Optional) Generic Webhook URL.                                           | No        | Leave empty to disable generic Webhook notifications.                                                                                               |
    | `WEBHOOK_METHOD`   | (Optional) Webhook request method.                                        | No        | Supports `GET` or `POST`. Defaults to `POST`.                                                                                                   |
    | `WEBHOOK_HEADERS`  | (Optional) Custom headers for Webhook.                                     | No        | Valid JSON string, e.g., `'{"Authorization": "Bearer xxx"}'`.                                                                                    |
    | `WEBHOOK_CONTENT_TYPE`| (Optional) Content type for POST requests.                             | No        | Supports `JSON` or `FORM`. Defaults to `JSON`.                                                                                                    |
    | `WEBHOOK_QUERY_PARAMETERS`| (Optional) Query parameters for GET requests.                            | No        | JSON string, supports `{{title}}` and `{{content}}` placeholders.                                                                              |
    | `WEBHOOK_BODY`     | (Optional) Request body for POST requests.                             | No        | JSON string, supports `{{title}}` and `{{content}}` placeholders.                                                                              |
    | `LOGIN_IS_EDGE`   | Use Edge browser for login and scraping.                                 | No        | Defaults to `false` (Chrome/Chromium).                                                                                                              |
    | `PCURL_TO_MOBILE` | Convert computer links to mobile links in notifications.                  | No        | Defaults to `true`.                                                                                                                                  |
    | `RUN_HEADLESS`    | Run the browser in headless mode.                                          | No        | Defaults to `true`. Set to `false` for local debugging when encountering CAPTCHAs. **Must be `true` for Docker deployments.**                       |
    | `AI_DEBUG_MODE`   | Enable AI debugging mode.                                            | No        | Defaults to `false`. Prints detailed AI request and response logs to the console.                                                                    |
    | `SKIP_AI_ANALYSIS` | Skip AI analysis and send notifications directly.                     | No        | Defaults to `false`. If set to `true`, all fetched items will be notified without AI analysis.                                                   |
    | `ENABLE_THINKING` | Enable enable_thinking parameter.                     | No        | Defaults to `false`. Some AI models require this parameter, while others don't support it.  If you encounter the error "Invalid JSON payload received. Unknown name "enable_thinking"", try setting this to `false`.                                                                                |
    | `SERVER_PORT`     | Web UI service port.                                               | No        | Defaults to `8000`.                                                                                                                                  |
    | `WEB_USERNAME`    | Web UI login username.                                                  | No        | Defaults to `admin`.  **Change this in production.**                                                                                            |
    | `WEB_PASSWORD`    | Web UI login password.                                                  | No        | Defaults to `admin123`.  **Change this to a strong password in production!**                                                                    |

    > 💡 **Debugging Tip:**  If you experience 404 errors when configuring your AI API, try using APIs from Alibaba Cloud or Volcano Engine for initial testing to ensure basic functionality before switching to other providers. Some providers may have compatibility issues or require special configuration.

    > 🔐 **Security Warning:** The Web interface uses Basic Authentication.  The default username and password are `admin` / `admin123`. **Change these to strong credentials in production!**

2.  **Get Login Status (Important!)**: The scraper requires valid login credentials. We strongly recommend using the Web UI for this:

    **Recommended Method: Web UI Update**

    1.  Skip ahead to Step 3, then launch the web service.
    2.  Open the Web UI, navigate to the "System Settings" page.
    3.  Find "Login State File" and click the "Manual Update" button.
    4.  Follow the instructions in the pop-up:
        *   Install the [Xianyu Login State Extractor extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        *   Open and log in to the Xianyu website.
        *   After successful login, click the extension icon in the browser toolbar.
        *   Click the "Extract Login State" button.
        *   Click "Copy to Clipboard" to copy the information.
        *   Paste the copied content into the Web UI and save it.

    This method avoids running a GUI-based program on your server and is the most convenient.

    **Alternative Method: Run Login Script** (If you have a local or GUI server):

    ```bash
    python login.py
    ```

    This will open a browser window; log in by scanning the QR code with the Xianyu app.  Upon successful login, the script will close and generate an `xianyu_state.json` file in the project root.

### Step 3: Start the Web Service

```bash
python web_server.py
```

### Step 4: Start Monitoring!

1.  Open your browser and go to `http://127.0.0.1:8000`.
2.  Go to the "Task Management" page and click "Create New Task".
3.  Describe your desired item in natural language (e.g., "I want to buy a Sony A7M4 camera, 95% new or better, budget under 13,000 yuan, shutter count below 5000"), and fill in the task name and keywords.
4.  Click "Create." The AI will generate complex filtering criteria.
5.  Return to the main interface, add a schedule, or directly start the task to begin automated monitoring!

## Docker Deployment (Recommended)

Docker provides a standardized way to deploy and run applications with all their dependencies.

### Step 1: Environment Preparation (Similar to Local Setup)

1.  **Install Docker:** Ensure you have [Docker Engine](https://docs.docker.com/engine/install/) installed.
2.  **Clone and Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env` File:** Follow the instructions in the **[Getting Started](#getting-started)** section to create and populate the `.env` file in the project root.
4.  **Get Login Status (Critical!)**: You cannot perform QR code login within the Docker container. **After starting the container**, configure your login status via the Web UI:

    1.  (On your host machine) Run `docker-compose up -d` to start the service.
    2.  Open the Web UI in your browser: `http://127.0.0.1:8000`.
    3.  Navigate to the "System Settings" page, and click the "Manual Update" button.
    4.  Follow the instructions in the pop-up:
        *   Install the [Xianyu Login State Extractor extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
        *   Open and log in to the Xianyu website.
        *   After successful login, click the extension icon in the browser toolbar.
        *   Click the "Extract Login State" button.
        *   Click "Copy to Clipboard" to copy the information.
        *   Paste the copied content into the Web UI and save it.

> ℹ️ **About Python Version:** The Docker image uses Python 3.11, as defined in the Dockerfile, so you don't need to worry about local Python version compatibility.

### Step 2: Run the Docker Container

The project includes a `docker-compose.yaml` file. Use `docker-compose` to manage the container (more convenient than `docker run`).

In the project root, run:

```bash
docker-compose up --build -d
```

This builds and starts the service in detached mode. `docker-compose` uses the `.env` file and `docker-compose.yaml` to configure the container.

If you encounter network issues inside the container, troubleshoot your proxy settings or container network configuration to ensure the container can access external networks.

> ⚠️ **OpenWrt Deployment Note:** If deploying on an OpenWrt router, you might experience DNS resolution issues.  The default network created by Docker Compose might not inherit OpenWrt's DNS settings correctly.  If you get `ERR_CONNECTION_REFUSED`, check your container network settings and potentially manually configure DNS or adjust the network mode to ensure external network access.

### Step 3: Access and Manage

*   **Access Web UI:** Open `http://127.0.0.1:8000` in your browser.
*   **View Real-Time Logs:** `docker-compose logs -f`
*   **Stop the Container:** `docker-compose stop`
*   **Start a Stopped Container:** `docker-compose start`
*   **Stop and Remove the Container:** `docker-compose down`

## Web UI Feature Overview

*   **Task Management:**
    *   **AI Task Creation:** Generate monitoring tasks and AI analysis criteria by describing your needs in natural language.
    *   **Visual Editing & Control:**  Modify task parameters (keywords, prices, schedules) directly in the table, and start/stop/delete tasks independently.
    *   **Scheduled Execution:** Configure Cron expressions for automated, periodic task runs.
*   **Result Viewing:**
    *   **Card View:**  Clear display of items matching your criteria in a card format, with images.
    *   **Smart Filtering & Sorting:** Filter by AI recommendation status, sort by crawl time, publish time, price, and more.
    *   **Deep Detail:**  Click to view complete scraped data and detailed AI analysis results in JSON format.
*   **Running Logs:**
    *   **Real-Time Log Stream:** View detailed, real-time logs in the web interface to track progress and troubleshoot issues.
    *   **Log Management:**  Automatic refresh, manual refresh, and one-click log clearing.
*   **System Settings:**
    *   **Status Check:**  One-click check of the `.env` configuration and login status to verify key dependencies.
    *   **Prompt Editing:**  Edit and save the `prompt` file used for AI analysis directly within the web interface to adjust the AI's reasoning logic in real-time.

## Authentication

### Authentication Configuration

The Web interface is secured with Basic Authentication to restrict access to authorized users and the API.

#### Configuration

Set authentication credentials in the `.env` file:

```bash
# Web service authentication configuration
WEB_USERNAME=admin
WEB_PASSWORD=admin123
```

#### Default Credentials

If authentication credentials are not set in `.env`, the system uses the following defaults:

-   Username: `admin`
-   Password: `admin123`

**⚠️ Important: Change the default password in production!**

#### Scope of Authentication

-   **Requires Authentication:** All API endpoints, Web interface, and static resources
-   **No Authentication Required:** Health check endpoint (`/health`)

#### Usage

1.  **Browser Access:** A login dialog will appear when accessing the Web interface.
2.  **API Calls:**  Include Basic Authentication information in the request headers.
3.  **Frontend JavaScript:**  Handles authentication automatically; no modifications are needed.

#### Security Recommendations

1.  Change the default password to a strong password.
2.  Use HTTPS in production.
3.  Change authentication credentials regularly.
4.  Limit access to the allowed IP range using a firewall.

Refer to [AUTH_README.md](AUTH_README.md) for detailed configuration instructions.

## Frequently Asked Questions (FAQ)

We have a comprehensive FAQ document covering various topics, including environment setup, AI configuration, and anti-scraping strategies.

👉 **[Click here to view the FAQ (FAQ.md)](FAQ.md)**

## Acknowledgements

This project was developed with inspiration from the following excellent projects:

-   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

and thanks to the script contributions from the LinuxDo related personnel.

-   [@jooooody](https://linux.do/u/jooooody/summary)

And thanks to the [LinuxDo](https://linux.do/) community.

Also thanks to ClaudeCode/ModelScope/Gemini and other models/tools, freeing hands and experiencing the joy of Vibe Coding.

## Experience

90%+ of the code for this project was generated by AI, including the PRs involved in the ISSUE.

The terrifying thing about Vibe Coding is that if you don't participate too much in the project construction, don't review the AI-generated code in detail, and don't think about why the AI is writing it this way, blindly passing test cases to verify functionality will only lead to the project becoming a black box.

Similarly, when using AI to code review AI-generated code, it's like using AI to verify whether another AI's answer is AI, falling into a self-proving dilemma, so AI can help with analysis, but it shouldn't be the arbiter of truth.

AI is omnipotent and can help developers solve 99% of coding problems. However, AI is also not omnipotent, and every problem solved requires developers to verify and think about it. AI is an aid, and the content AI produces can only be an aid as well.

## ⚠️ Important Notes

-   Comply with Xianyu's user agreement and robots.txt rules. Avoid excessively frequent requests to avoid server overload or account restrictions.
-   This project is for educational and technical research purposes only. Do not use it for illegal activities.
-   This project is released under the [MIT License](LICENSE), provided "as is" without any warranty.
-   The project author and contributors are not liable for any direct, indirect, incidental, or special damages resulting from using this software.
-   Please refer to the [DISCLAIMER.md](DISCLAIMER.md) file for more detailed information.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)