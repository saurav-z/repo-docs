# AI-Powered Goofish Monitor: Real-time, Smart, and Customizable 闲鱼 (Goofish) Item Tracking

This project leverages Playwright and AI for intelligent monitoring and analysis of 闲鱼 (Goofish) listings, with a user-friendly web interface.  [View the original repository on GitHub](https://github.com/dingyufei615/ai-goofish-monitor).

**Key Features:**

*   ✅ **Web UI for Easy Management:** Intuitive web interface for task management, AI configuration, real-time log viewing, and result filtering.
*   💬 **AI-Driven Task Creation:**  Generate complex monitoring tasks with natural language descriptions of your desired items.
*   🚀 **Multi-Task Concurrency:** Monitor multiple keywords simultaneously with independent, non-interfering tasks via `config.json`.
*   ⚡️ **Real-time Processing:**  Immediate analysis of new listings, eliminating batch processing delays.
*   🧠 **Deep AI Analysis:** Integrated multimodal LLMs (like GPT-4o) analyze item descriptions, images, and seller profiles for precise filtering.
*   ⚙️ **Highly Customizable:** Configure individual keywords, price ranges, filtering criteria, and AI analysis prompts for each task.
*   🔔 **Instant Notifications:**  Receive notifications on your phone or desktop via ntfy.sh, Enterprise WeChat group bots, and Bark.
*   📅 **Scheduled Task Execution:**  Utilize cron expressions for automated, timed task execution.
*   🐳 **Docker Deployment:**  Simplified deployment with `docker-compose` for rapid, standardized containerization.
*   🛡️ **Robust Anti-Scraping:**  Employs realistic browser behavior with randomized delays for stable operation.

## Core Functionality Explained

The system efficiently tracks items as described in the following process flow:

1.  **Initiate Monitoring:** Start a monitoring task.
2.  **Search for Items:** The system searches for items based on predefined criteria.
3.  **New Item Check:** Identifies new listings that match the criteria.
4.  **Detailed Data Retrieval:** Grabs item details and seller information.
5.  **Image Download:** Retrieves item images.
6.  **AI Analysis:** Evaluates the items based on AI.
7.  **Recommendation Check:** Determines if the AI recommends the item.
8.  **Notification Delivery:** Sends alerts when a match occurs.
9.  **Data Storage:** Saves item information if a match is found.
10. **Repeat Cycle:** Returns to the item search phase to continue monitoring.

## Quick Start

### Prerequisites
1.  **Python Version:** Requires Python 3.10 or higher. (May lead to dependency errors on older Python versions.)

2.  **Clone and install dependencies:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    pip install -r requirements.txt
    ```

### Configuration:

1.  **Set Environment Variables:**  Configure your `.env` file. (Create a copy of `.env.example`)

    **Create the `.env` file:**
    ```bash
    # For Windows users
    copy .env.example .env

    # For Linux or macOS users
    cp .env.example .env
    ```

    Environment Variable Details:

    | Variable | Description | Required? | Notes |
    | --------- | ----------- | --------- | ----- |
    | `OPENAI_API_KEY` | AI API Key | Yes | |
    | `OPENAI_BASE_URL` | AI API Endpoint | Yes | Compatible with OpenAI format. |
    | `OPENAI_MODEL_NAME` | Your AI Model Name | Yes | **Required:** Use a multimodal model (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`). |
    | `PROXY_URL` | Optional Proxy URL (HTTP/S) | No | Use `http://` or `socks5://`. |
    | `NTFY_TOPIC_URL` | ntfy.sh Topic URL | No | For sending notifications. |
    | `GOTIFY_URL` | Gotify Service Address | No |  |
    | `GOTIFY_TOKEN` | Gotify Application Token | No |  |
    | `BARK_URL` | Bark Push Address | No | For sending notifications. |
    | `WX_BOT_URL` | Enterprise WeChat Bot Webhook URL | No |  |
    | `WEBHOOK_URL` | Generic Webhook URL | No |  |
    | `WEBHOOK_METHOD` | Webhook Request Method | No |  `GET` or `POST`. Default is `POST`. |
    | `WEBHOOK_HEADERS` | Custom Webhook Request Headers | No | Valid JSON String. |
    | `WEBHOOK_CONTENT_TYPE` | POST Request Content Type | No | `JSON` or `FORM`. Default is `JSON`. |
    | `WEBHOOK_QUERY_PARAMETERS` | GET Request Query Parameters | No | JSON String. Supports `{{title}}` and `{{content}}`. |
    | `WEBHOOK_BODY` | POST Request Body | No | JSON String. Supports `{{title}}` and `{{content}}`. |
    | `LOGIN_IS_EDGE` | Login and Crawl with Edge Browser | No | Default is `false`. |
    | `PCURL_TO_MOBILE` | Convert PC links to Mobile links | No | Default is `true`. |
    | `RUN_HEADLESS` | Run Browser in Headless Mode | No | Default is `true`.  **Required to be `true` for Docker.** |
    | `AI_DEBUG_MODE` | Enable AI Debug Mode | No | Default is `false`.  |
    | `SKIP_AI_ANALYSIS` | Skip AI Analysis | No | Default is `false`. |
    | `ENABLE_THINKING` | Enable "enable_thinking" parameter for AI models | No | Default is `false`. |
    | `SERVER_PORT` | Web UI Port | No | Default is `8000`. |
    | `WEB_USERNAME` | Web UI Username | No | Default is `admin`.  **Change in production!** |
    | `WEB_PASSWORD` | Web UI Password | No | Default is `admin123`.  **Change in production!** |

    > **Troubleshooting API Errors:** If you get 404 errors, test your configuration with services like Alibaba Cloud or Volcano Engine, then try other providers.

    > **Security Note:** The Web UI uses Basic Auth.  Default credentials are `admin`/`admin123`.  **Change these in production!**

2.  **Get Login Status:** The crawler needs valid login credentials.  There are two ways to get the login:

    **Recommended: Web UI Update**

    1.  Skip to step 3 to launch the Web server.
    2.  Go to "System Settings" within the Web UI.
    3.  Click the "Manually Update" button next to "Login Status File."
    4.  Follow the instructions in the pop-up:
        *   Install the 闲鱼 login state extraction extension in Chrome.
        *   Log in to the 闲鱼 website using Chrome.
        *   Click the extension icon to get the login info.
        *   Click "Copy to Clipboard."
        *   Paste the copied data into the Web UI and save.

    **Alternative: Run login.py (If you can run a browser locally):**

    ```bash
    python login.py
    ```

    A browser window will open, requiring you to scan a QR code with your 闲鱼 app to log in. A `xianyu_state.json` file will be created in the project root upon successful login.

### Run the Web Service
```bash
python web_server.py
```
### Start Monitoring
1.  Open your browser to `http://127.0.0.1:8000`.
2.  Go to the "Task Management" page and click "Create New Task."
3.  Describe your requirements using natural language, such as "Looking for a Sony A7M4 camera, minimum condition 95% new, budget under 13000 yuan, shutter count below 5000."  Provide a task name and keywords.
4.  The AI will automatically set up advanced analysis criteria.
5.  Start the task by clicking Start, or by setting a schedule.

## Docker Deployment (Recommended)

### Docker Prerequisites

1.  **Install Docker:** Ensure Docker Engine is installed.

2.  **Clone and Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create .env:** Refer to the [Quick Start](#quick-start) section and configure the `.env` file.

4.  **Get Login Status (Important for Docker):** You must configure login via the Web UI after starting the container:

    1.  Run `docker-compose up -d` on your host machine.
    2.  Open `http://127.0.0.1:8000` in your browser.
    3.  Go to "System Settings" and click "Manually Update."
    4.  Follow the Web UI instructions for extracting and pasting the login state.

### Run Docker Containers:

```bash
docker-compose up --build -d
```

### Manage your container:

-   **Access Web UI:**  `http://127.0.0.1:8000`
-   **View Logs:** `docker-compose logs -f`
-   **Stop Container:** `docker-compose stop`
-   **Start Container:** `docker-compose start`
-   **Stop and Remove Container:** `docker-compose down`

## Web UI Features

*   **Task Management:**
    *   AI-powered Task Creation: Generate monitoring tasks using natural language.
    *   Visual Editing: Modify task parameters and control tasks directly.
    *   Scheduled Execution: Automate tasks using cron expressions.
*   **Results Viewing:**
    *   Card-based Display: Displays item info as cards.
    *   Smart Filtering and Sorting: Easily filter by AI recommendations and sort.
    *   Detailed Views: See item data, AI analysis.
*   **Real-time Logs:**
    *   Detailed Logs: View running logs.
    *   Log Management: Auto/manual refresh and clear options.
*   **System Settings:**
    *   Health Checks: Verify configuration.
    *   Prompt Editing: Edit AI prompts directly.

## FAQs and Disclaimer

Find the detailed FAQ [here](FAQ.md).  The project adheres to the [MIT License](LICENSE) and is provided "as is." See the [DISCLAIMER.md](DISCLAIMER.md) file for limitations of liability.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)