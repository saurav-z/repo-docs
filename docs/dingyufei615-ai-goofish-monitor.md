# AI-Powered Goofish (Xianyu) Monitor: Smart Item Tracking & Analysis

Tired of missing out on the perfect second-hand find? This project offers an intelligent Xianyu (闲鱼) monitoring solution, leveraging AI for advanced item analysis and real-time alerts.  **[Check out the original repo for more details and to get started!](https://github.com/dingyufei615/ai-goofish-monitor)**

## Key Features:

*   **Intuitive Web UI:** Manage tasks, edit AI criteria, view logs, and filter results effortlessly.
*   **AI-Driven Task Creation:** Describe your desired item in natural language, and the AI will generate a sophisticated monitoring task.
*   **Concurrent Multi-Tasking:** Monitor multiple keywords and criteria simultaneously, all running independently.
*   **Real-time Stream Processing:** Receive immediate analysis and alerts as new items appear.
*   **Deep AI Analysis:** Utilize multimodal LLMs (like GPT-4o) to analyze item descriptions, images, and seller profiles for accurate filtering.
*   **Highly Customizable:** Configure individual tasks with unique keywords, price ranges, filters, and AI prompts.
*   **Instant Notifications:** Get alerted via ntfy.sh, WeChat Enterprise Bot, or Bark.
*   **Scheduled Task Execution:** Employ Cron expressions for automated, periodic monitoring.
*   **Docker-Friendly Deployment:** Simplify setup with provided Docker Compose configuration.
*   **Robust Anti-Scraping Measures:** Simulate human behavior, including random delays, to improve stability.

## Web UI Screenshots

**Task Management**
![img.png](static/img.png)

**Monitoring**
![img_1.png](static/img_1.png)

**Notification**
![img_2.png](static/img_2.png)

## Getting Started (Web UI Recommended)

### Step 1: Environment Setup

>   ⚠️ **Python Version Requirement:** Use Python 3.10 or higher for local development. Older versions may cause dependency installation or runtime errors (e.g., `ModuleNotFoundError: No module named 'PIL'`).

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

1.  **Configure Environment Variables:** Copy `.env.example` to `.env` and customize the settings.

    **Command for Windows:**

    ```cmd
    copy .env.example .env
    ```

    **Command for Linux/MacOS:**

    ```bash
    cp .env.example .env
    ```

    Available environment variables:

    | Variable            | Description                                   | Required? | Notes                                                                                        |
    | :------------------ | :-------------------------------------------- | :-------- | :------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`    | Your AI model provider's API Key.             | Yes       | May be optional for some local or proxy services.                                             |
    | `OPENAI_BASE_URL`   | AI model API endpoint (OpenAI-compatible).    | Yes       | Enter the base path of the API, e.g., `https://ark.cn-beijing.volces.com/api/v3/`.         |
    | `OPENAI_MODEL_NAME` | The specific model you want to use.           | Yes       | **Must** be a multimodal model (supports image analysis), like `doubao-seed-1-6-250615`,  `gemini-2.5-pro`. |
    | `PROXY_URL`         | (Optional) HTTP/S proxy for bypassing filters. | No        | Supports `http://` and `socks5://`, e.g., `http://127.0.0.1:7890`.                         |
    | `NTFY_TOPIC_URL`    | (Optional) ntfy.sh topic URL for notifications. | No        | Leave blank to disable ntfy notifications.                                                  |
    | `GOTIFY_URL`        | (Optional) Gotify service address.             | No        | e.g., `https://push.example.de`.                                                               |
    | `GOTIFY_TOKEN`      | (Optional) Gotify app token.                  | No        |                                                                                              |
    | `BARK_URL`          | (Optional) Bark push address.                 | No        | e.g., `https://api.day.app/your_key`.  Leave blank to disable Bark notifications.              |
    | `WX_BOT_URL`        | (Optional) WeChat Enterprise Bot webhook.     | No        | Leave blank to disable WeChat notifications.                                                  |
    | `WEBHOOK_URL`       | (Optional) Generic Webhook URL.               | No        | Leave blank to disable generic Webhook notifications.                                             |
    | `WEBHOOK_METHOD`    | (Optional) Webhook request method.             | No        | Supports `GET` or `POST`, defaults to `POST`.                                                 |
    | `WEBHOOK_HEADERS`   | (Optional) Custom Webhook headers.             | No        | Must be a valid JSON string, e.g., `'{"Authorization": "Bearer xxx"}'`.                       |
    | `WEBHOOK_CONTENT_TYPE`| (Optional) POST request content type. | No | Supports `JSON` or `FORM`, defaults to `JSON`. |
    | `WEBHOOK_QUERY_PARAMETERS`| (Optional) GET request query parameters.  | No       | JSON string, supporting `{{title}}` and `{{content}}` placeholders.                           |
    | `WEBHOOK_BODY`      | (Optional) POST request body.                | No        | JSON string, supporting `{{title}}` and `{{content}}` placeholders.                           |
    | `LOGIN_IS_EDGE`     | Use Edge browser for login and scraping.      | No        | Defaults to `false` (Chrome/Chromium).                                                        |
    | `PCURL_TO_MOBILE`   | Convert PC item links to mobile links.        | No        | Defaults to `true`.                                                                         |
    | `RUN_HEADLESS`      | Run browser in headless mode.                 | No        | Defaults to `true`. Set to `false` for local debugging if you encounter CAPTCHAs. **Must be `true` for Docker deployment.** |
    | `AI_DEBUG_MODE`     | Enable AI debug mode.                        | No        | Defaults to `false`. Prints detailed AI request/response logs to the console.                     |
    | `SERVER_PORT`       | Web UI server port.                         | No        | Defaults to `8000`.                                                                          |

    >   💡 **Debugging Tip:** If you encounter 404 errors when configuring the AI API, test with an API from Alibaba Cloud or Volcano Engine first to ensure basic functionality. Some API providers may have compatibility issues or require specific configurations.

2.  **Obtain Login State (Important!)**:  Provide valid login credentials for the scraper.  Use the Web UI for the easiest method:

    **Recommended Method: Update via Web UI**

    1.  Skip this step and proceed to Step 3 to start the web server.
    2.  Open the Web UI and navigate to the "System Settings" page.
    3.  Find "Login State File" and click the "Manual Update" button.
    4.  Follow the instructions in the popup:
        -   Install the [Xianyu Login State Extractor extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in your Chrome browser.
        -   Open and log in to Xianyu's official website.
        -   After successful login, click the extension icon in your browser toolbar.
        -   Click the "Extract Login State" button to obtain login information.
        -   Click the "Copy to Clipboard" button.
        -   Paste the copied content into the Web UI and save.

    This method is the most convenient as it doesn't require running a GUI program on the server.

    **Alternative Method: Run the Login Script (if you have a local GUI environment):**

    ```bash
    python login.py
    ```

    This will open a browser window.  Use the **Xianyu App on your phone to scan the QR code** to log in. Upon successful login, the program closes, and a `xianyu_state.json` file is created in the project root.

### Step 3: Start the Web Server

Once configured, launch the web server:

```bash
python web_server.py
```

### Step 4: Start Monitoring

1.  Open your browser and go to `http://127.0.0.1:8000`.
2.  In the "Task Management" page, click "Create New Task".
3.  In the window, describe your purchase needs in natural language (e.g., "I want to buy a Sony A7M4 camera, 95% new or better, budget under 13,000, shutter count below 5000"), and fill in task name, keywords, etc.
4.  Click Create, and the AI will generate the criteria.
5.  Go back to the main interface, schedule the task or click start to begin monitoring!

## Docker Deployment (Recommended)

Docker simplifies deployment with standardized containers.

### Step 1: Environment Preparation (Similar to Local Setup)

1.  **Install Docker:** Ensure [Docker Engine](https://docs.docker.com/engine/install/) is installed.

2.  **Clone the project and configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env` File:** Create and populate your `.env` file based on the **[Getting Started](#getting-started-web-ui-recommended)** instructions.

4.  **Obtain Login State (Critical!)**: You cannot scan a QR code within the Docker container.  **After** starting the container, use the Web UI to set the login state:
    1.  (On the host machine) Run `docker-compose up -d` to start the service.
    2.  Open your browser and go to `http://127.0.0.1:8000`.
    3.  Navigate to the "System Settings" page and click the "Manual Update" button.
    4.  Follow the instructions in the popup:
        -   Install the [Xianyu Login State Extractor extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in your Chrome browser.
        -   Open and log in to Xianyu's official website.
        -   After successful login, click the extension icon in your browser toolbar.
        -   Click the "Extract Login State" button to obtain login information.
        -   Click the "Copy to Clipboard" button.
        -   Paste the copied content into the Web UI and save.

>   ℹ️ **Regarding Python Version:**  Docker deployment uses the Python 3.11 version specified in the Dockerfile, so you don't need to worry about local Python version compatibility.

### Step 2: Run Docker Container

The project includes `docker-compose.yaml`. We recommend using `docker-compose` for easier container management.

Run the following command in the project root:

```bash
docker-compose up --build -d
```

This starts the service in the background.  `docker-compose` uses `.env` and `docker-compose.yaml` to build and launch the container.

If you encounter network issues within the container, troubleshoot or use a proxy.

>   ⚠️ **OpenWrt Deployment Notes:** Deploying on an OpenWrt router may cause DNS resolution problems. This is because the default network created by Docker Compose might not inherit OpenWrt's DNS settings correctly.  If you encounter `ERR_CONNECTION_REFUSED` errors, check your container's network configuration, and potentially manually configure DNS or adjust the network mode to ensure the container can access the external network.

### Step 3: Access and Manage

-   **Access Web UI:** Open `http://127.0.0.1:8000` in your browser.
-   **View Real-time Logs:** `docker-compose logs -f`
-   **Stop Container:** `docker-compose stop`
-   **Start Stopped Container:** `docker-compose start`
-   **Stop and Remove Container:** `docker-compose down`

## Web UI Feature Overview

*   **Task Management:**
    *   **AI-Powered Task Creation:** Generate monitoring tasks with AI, using natural language descriptions.
    *   **Visual Editing & Control:** Edit task parameters (keywords, price, scheduling, etc.) and independently start/stop and delete tasks.
    *   **Cron Scheduling:** Set Cron expressions for automated periodic execution.
*   **Result Viewing:**
    *   **Card-Based Display:** Display matching items with images and descriptions.
    *   **Smart Filtering & Sorting:** Filter for AI-recommended items and sort by crawl time, publish time, price, etc.
    *   **Detailed Information:** Access complete scraped data and AI analysis results.
*   **Running Logs:**
    *   **Real-Time Log Stream:** View detailed real-time logs for progress tracking and troubleshooting.
    *   **Log Management:** Support auto-refresh, manual refresh, and one-click log clearing.
*   **System Settings:**
    *   **Status Checks:** Check the key dependencies such as `.env` configuration and login state.
    *   **Prompt Online Editing:** Edit and save the AI analysis prompt files directly in the web UI.

## Workflow

The diagram below illustrates the core processing logic of a single monitoring task.  `web_server.py` starts one or more of these tasks based on user interaction or scheduling.

```mermaid
graph TD
    A[Start Monitoring Task] --> B[Task: Search Items];
    B --> C{New Item Found?};
    C -- Yes --> D[Fetch Item Details & Seller Info];
    D --> E[Download Item Images];
    E --> F[Call AI for Analysis];
    F --> G{AI Recommends?};
    G -- Yes --> H[Send Notifications];
    H --> I[Save Record to JSONL];
    G -- No --> I;
    C -- No --> J[Page/Wait];
    J --> B;
    I --> C;
```

## Frequently Asked Questions (FAQ)

1.  **Q:  I get a  `'gbk' codec can't encode character` error?**
    *   **A:**  This is a Windows-specific encoding issue. The code and logs use UTF-8.
    *   **Solution:** Before running the Python script, set the environment variable to enforce UTF-8.  In PowerShell or CMD:

        ```bash
        set PYTHONUTF8=1
        python spider_v2.py
        ```

        Or, use `chcp 65001` to change the active code page to UTF-8.

2.  **Q:  `login.py` shows  `'playwright install'`  needed?**
    *   **A:**  This means Playwright is missing the required browser files. Ensure dependencies are correctly installed via `requirements.txt`:

        ```bash
        pip install -r requirements.txt
        ```

        If the issue persists, try installing Chromium:

        ```bash
        playwright install chromium
        ```

3.  **Q:  "Request timed out" or "Connection error" errors?**
    *   **A:**  Usually a network issue, meaning your server can't connect to  `OPENAI_BASE_URL`.  Check:
        *   Server network connectivity.
        *   If in mainland China, consider a network proxy for accessing foreign AI services. You can configure `PROXY_URL` in `.env`.
        *   Verify that `OPENAI_BASE_URL` is correct and that the service is running.

4.  **Q: AI model doesn't support image analysis?**
    *   **A:**  The project's core feature is multimodal analysis.  **You MUST** select an AI model supporting image recognition (Vision / Multi-modal).  Change  `OPENAI_MODEL_NAME`  in `.env`  to a model like  `gpt-4o`, `gemini-1.5-pro`,  `deepseek-v2`,  `qwen-vl-plus`, etc.

5.  **Q: Deploy on Synology NAS via Docker?**
    *   **A:**  Yes.  Follow the standard Docker deployment steps with these adjustments:
        1.  Complete  `login.py`  on your computer (not the Synology) to generate  `xianyu_state.json`.
        2.  Upload the project folder (with  `.env`  and  `xianyu_state.json`) to a Synology directory.
        3.  In Synology's Container Manager, use  `docker-compose up -d` (via SSH or Task Scheduler), ensuring the volume mappings in  `docker-compose.yaml`  point to your Synology project folder.

6.  **Q:  Configure Gemini / Qwen / Grok / other non-OpenAI models?**
    *   **A:**  The project supports any model with an OpenAI-compatible API. Key is configuring the  `.env`  variables correctly:
        *   `OPENAI_API_KEY`:  Your API Key.
        *   `OPENAI_BASE_URL`:  API-Compatible Endpoint from your provider. Consult your model's documentation; usually, it's  `https://api.your-provider.com/v1`  (without  `/chat/completions`  at the end).
        *   `OPENAI_MODEL_NAME`:  Your specific model name. Needs to support image recognition, e.g., `gemini-2.5-flash`.
    *   **Example:**  If your provider says the Completions endpoint is `https://xx.xx.com/v1/chat/completions`, then  `OPENAI_BASE_URL`  should be  `https://xx.xx.com/v1`.

7.  **Q:  Getting flagged as "abnormal traffic" or requiring CAPTCHAs?**
    *   **A:** This is Xianyu's anti-scraping mechanism.  Try:
        *   **Disable Headless Mode:** Set  `RUN_HEADLESS=false`  in  `.env`. The browser runs with a UI; complete CAPTCHAs manually.
        *   **Reduce Monitoring Frequency:** Limit concurrent tasks.
        *   **Use a Clean Network Environment:** Frequent scraping can lead to IP flagging.

8.  **Q:  pyzbar installation fails on Windows?**
    *   **A:**  pyzbar requires the zbar dynamic link library on Windows.
    *   **Solutions (Windows):**
        *   **Method 1 (Recommended):** Use Chocolatey:

            ```cmd
            choco install zbar
            ```

        *   **Method 2:** Manual Download and PATH addition:
            1.  Download the correct  `libzbar-64.dll`  version from [zbar releases](https://github.com/NaturalHistoryMuseum/pyzbar/releases).
            2.  Place the file in your Python installation directory or add it to your system PATH.
        *   **Method 3:**  Use conda:

            ```cmd
            conda install -c conda-forge zbar
            ```

    *   **Linux Users:** Install system packages:

        ```bash
        # Ubuntu/Debian
        sudo apt-get install libzbar0
        
        # CentOS/RHEL
        sudo yum install zbar
        
        # Arch Linux
        sudo pacman -S zbar
        ```

9.  **Q:  `ModuleNotFoundError: No module named 'PIL'`  during  `login.py`  run?**
    *   **A:**  Often related to a low Python version or incomplete dependency installation. This project recommends Python 3.10+.
    *   **Solutions:**
        *   Ensure you're using Python 3.10+
        *   Reinstall dependencies:

            ```bash
            pip install -r requirements.txt
            ```

        *   If still a problem, install Pillow separately:

            ```bash
            pip install Pillow
            ```

10. **Q: 404 errors when configuring the AI API?**
    *   **A:** If you encounter 404 errors when configuring the AI API, test with an API from Alibaba Cloud or Volcano Engine first to ensure basic functionality. Some API providers may have compatibility issues or require specific configurations. Please check:
        - Verify that the `OPENAI_BASE_URL` address is correct, and that the service is running properly.
        - Check if the network connection is normal.
        - Confirm the API Key is correct and has access permissions.
        - Some API providers may need special header or parameter configurations. Please consult the official documentation.

## Acknowledgements

This project references the following excellent projects:

*   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

Also, thanks for the script contribution from the LinuxDo related friends.

- [@jooooody](https://linux.do/u/jooooody/summary)

And thanks to Aider and Gemini to liberate my hands, it is like flying when I write the code.

## Support & Sponsoring

If the project helps you, please consider buying me a coffee, thank you very much for your support!

<table>
  <tr>
    <td><img src="static/zfb_support.jpg" width="200" alt="Alipay" /></td>
    <td><img src="static/wx_support.png" width="200" alt="WeChat Pay" /></td>
  </tr>
</table>

## ⚠️ Important Notes

*   Adhere to Xianyu's user agreement and robots.txt. Avoid excessive requests to prevent server load or account restrictions.
*   This project is for learning and technical research only. Do not use it for illegal purposes.
*   This project is released under the [MIT License](LICENSE) "as is" without any warranty.
*   The project authors and contributors are not liable for any direct, indirect, incidental, or special damages or losses resulting from the use of this software.
*   For more details, refer to the [DISCLAIMER.md](DISCLAIMER.md) file.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)