# AI-Powered Xianyu (Goofish) Monitor: Real-time Monitoring & Smart Analysis

**Effortlessly track and analyze Xianyu (Goofish) listings with AI, complete with a user-friendly web interface and robust features. [View the original repository](https://github.com/dingyufei615/ai-goofish-monitor)**

## Key Features:

*   ✅ **Intuitive Web UI:** Manage tasks, edit AI prompts, view real-time logs, and filter results without command-line fuss.
*   🧠 **AI-Driven Task Creation:** Describe your desired item in natural language and let the AI create complex monitoring tasks.
*   🚀 **Multi-Task Concurrency:** Monitor multiple keywords simultaneously with independent, non-interfering tasks configured via `config.json`.
*   ⚡ **Real-Time Processing:** Instantly analyze new listings, eliminating batch processing delays.
*   💡 **Deep AI Analysis:** Leverage multimodal LLMs (like GPT-4o) to analyze item descriptions, images, and seller profiles for precise filtering.
*   ⚙️ **Highly Customizable:** Configure each task with unique keywords, price ranges, filters, and AI analysis prompts.
*   🔔 **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat Work bot, and [Bark](https://bark.day.app/) for matching items.
*   📅 **Scheduled Task Execution:** Utilize Cron expressions for automated, scheduled monitoring.
*   🐳 **Docker Deployment Ready:** Deploy quickly with a pre-configured `docker-compose` setup.
*   🛡️ **Robust Anti-Scraping Measures:** Simulate human behavior with random delays and user actions for stable performance.

## Web UI Screenshots:

*   **Task Management:**
    ![img.png](static/img.png)
*   **Monitoring Interface:**
    ![img_1.png](static/img_1.png)
*   **Notification Example:**
    ![img_2.png](static/img_2.png)

## Quick Start (Web UI Recommended):

The Web UI provides the best user experience.

### Step 1: Environment Setup

> ⚠️ **Python Version:** Python 3.10 or higher is recommended for local deployment and debugging. Older versions may cause dependency installation issues or runtime errors (e.g., `ModuleNotFoundError: No module named 'PIL'`).

Clone the project:

```bash
git clone https://github.com/dingyufei615/ai-goofish-monitor
cd ai-goofish-monitor
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Configuration

1.  **Configure Environment Variables:**  Copy `.env.example` to `.env` and modify its contents.

    **Windows:**

    ```cmd
    copy .env.example .env
    ```

    **Linux/macOS:**

    ```bash
    cp .env.example .env
    ```

    Available environment variables:

    | Variable            | Description                                                              | Required? | Notes                                                                                                                                                     |
    | :------------------ | :----------------------------------------------------------------------- | :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`    | Your AI model provider's API Key.                                         | Yes       | May be optional for some local or proxy services.                                                                                                     |
    | `OPENAI_BASE_URL`   | The base URL for the AI model's API, must be OpenAI-compatible.          | Yes       | Fill in the base path of the API, e.g., `https://ark.cn-beijing.volces.com/api/v3/`.                                                                 |
    | `OPENAI_MODEL_NAME` | The specific model name you want to use.                               | Yes       | **Must** choose a multimodal model supporting image analysis, like `doubao-seed-1-6-250615`, `gemini-2.5-pro`, etc.                                    |
    | `PROXY_URL`         | (Optional) HTTP/S proxy for bypassing network restrictions.               | No        | Supports `http://` and `socks5://` formats. For example, `http://127.0.0.1:7890`.                                                                      |
    | `NTFY_TOPIC_URL`    | (Optional) [ntfy.sh](https://ntfy.sh/) topic URL for sending notifications. | No        | If left empty, no ntfy notifications will be sent.                                                                                                  |
    | `GOTIFY_URL`        | (Optional) Gotify service address.                                      | No        | For example, `https://push.example.de`.                                                                                                            |
    | `GOTIFY_TOKEN`      | (Optional) Gotify application token.                                    | No        |                                                                                                                                                         |
    | `BARK_URL`          | (Optional) [Bark](https://bark.day.app/) push address.                    | No        | For example, `https://api.day.app/your_key`. If empty, no Bark notifications will be sent.                                                                 |
    | `WX_BOT_URL`        | (Optional) Enterprise WeChat bot webhook address.                        | No        | If empty, no Enterprise WeChat notifications will be sent.                                                                                             |
    | `WEBHOOK_URL`       | (Optional) Generic Webhook URL.                                        | No        | If empty, no generic Webhook notifications will be sent.                                                                                               |
    | `WEBHOOK_METHOD`    | (Optional) Webhook request method.                                       | No        | Supports `GET` or `POST`, defaults to `POST`.                                                                                                          |
    | `WEBHOOK_HEADERS`   | (Optional) Custom Webhook request headers.                               | No        | Must be a valid JSON string, e.g., `'{"Authorization": "Bearer xxx"}'`.                                                                             |
    | `WEBHOOK_CONTENT_TYPE`| (Optional) Content type for POST requests.                              | No        | Supports `JSON` or `FORM`, defaults to `JSON`.                                                                                                         |
    | `WEBHOOK_QUERY_PARAMETERS`| (Optional) Query parameters for GET requests.                       | No        | JSON string, supports `{{title}}` and `{{content}}` placeholders.                                                                                     |
    | `WEBHOOK_BODY`      | (Optional) POST request body.                                           | No        | JSON string, supports `{{title}}` and `{{content}}` placeholders.                                                                                     |
    | `LOGIN_IS_EDGE`     | Whether to use the Edge browser for login and scraping.                  | No        | Defaults to `false`, using Chrome/Chromium.                                                                                                             |
    | `PCURL_TO_MOBILE`   | Whether to convert PC item links to mobile links in notifications.       | No        | Defaults to `true`.                                                                                                                                   |
    | `RUN_HEADLESS`      | Whether to run the browser in headless mode.                           | No        | Defaults to `true`. Set to `false` during local debugging if you encounter CAPTCHAs. **Must be `true` for Docker deployment.**                         |
    | `AI_DEBUG_MODE`     | Whether to enable AI debug mode.                                         | No        | Defaults to `false`. Enable to print detailed AI request and response logs to the console.                                                             |
    | `SERVER_PORT`       | The port the Web UI service runs on.                                      | No        | Defaults to `8000`.                                                                                                                                    |

    > 💡 **Debugging Tip:** If you encounter 404 errors when configuring the AI API, it is recommended to first use the API provided by Alibaba Cloud or Volcano Engine for debugging, and ensure that the basic functions are normal before trying other API providers. Some API providers may have compatibility issues or require special configurations.

2.  **Get Login State (Important!)**: You must provide valid login credentials for the scraper to access Xianyu.  We recommend using the Web UI:

    **Recommended: Using the Web UI to Update Login State**
    1.  Skip this step and proceed to Step 3 to start the Web service.
    2.  Open the Web UI and go to the **"System Settings"** page.
    3.  Find "Login State File" and click the **"Manual Update"** button.
    4.  Follow the instructions in the pop-up window:
        *   Install the [Xianyu Login State Extraction Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome on your PC.
        *   Open and log in to the Xianyu website.
        *   After successful login, click the extension icon in the browser toolbar.
        *   Click the "Extract Login State" button to get the login information.
        *   Click the "Copy to Clipboard" button.
        *   Paste the copied content into the Web UI and save it.

    This method is the most convenient, as it doesn't require running a program with a graphical interface on the server.

    **Alternative: Run the Login Script**
    If you can run programs locally or on a server with a desktop environment, you can use the traditional script method:

    ```bash
    python login.py
    ```

    This will open a browser window. Use your **mobile Xianyu App to scan the QR code** to log in. Upon success, the program will close automatically and generate an `xianyu_state.json` file in the project root.

### Step 3: Start the Web Service

Once ready, start the Web UI server:

```bash
python web_server.py
```

### Step 4: Start Monitoring

Open `http://127.0.0.1:8000` in your browser.

1.  Go to **"Task Management"** and click **"Create New Task"**.
2.  Describe your needs in natural language (e.g., "I want to buy a Sony A7M4 camera, 95% new or better, budget under 13,000, shutter count below 5000"), and fill in task name, keywords, etc.
3.  Click "Create". The AI will generate the analysis criteria.
4.  Go back to the main interface, set a schedule or click "Start" to begin automated monitoring!

## Docker Deployment (Recommended):

Docker enables quick, reliable, and consistent deployment by packaging the application and its dependencies.

### Step 1: Environment Setup (Similar to Local Deployment)

1.  **Install Docker**:  Ensure you have [Docker Engine](https://docs.docker.com/engine/install/) installed.
2.  **Clone and Configure**:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env` file**: Refer to **[Quick Start](#-快速开始-web-ui-推荐)** for creating and populating the `.env` file.
4.  **Get Login State (Critical!)**:  You cannot scan the QR code to log in inside the Docker container. Set the login state **after** starting the container via the Web UI:
    1.  (On the host machine) Run `docker-compose up -d` to start the service.
    2.  Open `http://127.0.0.1:8000` in your browser.
    3.  Go to the **"System Settings"** page, and click the **"Manual Update"** button.
    4.  Follow the instructions in the pop-up window:
        *   Install the [Xianyu Login State Extraction Extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome on your PC.
        *   Open and log in to the Xianyu website.
        *   After successful login, click the extension icon in the browser toolbar.
        *   Click the "Extract Login State" button to get the login information.
        *   Click the "Copy to Clipboard" button.
        *   Paste the copied content into the Web UI and save it.

> ℹ️ **Regarding Python Version:** When deploying with Docker, the project uses Python 3.11, specified in the Dockerfile, so you don't need to worry about local Python version compatibility.

### Step 2: Run the Docker Container

The project includes `docker-compose.yaml`. Use `docker-compose` for easier container management.

Run this command in the project root:

```bash
docker-compose up --build -d
```

This starts the service in detached mode. `docker-compose` reads `.env` and `docker-compose.yaml` to create and start the container.

If you encounter network issues within the container, troubleshoot or use a proxy.

> ⚠️ **OpenWrt Deployment Notes**:  If deploying on an OpenWrt router, you might encounter DNS resolution problems. This is because the default network created by Docker Compose may not correctly inherit OpenWrt's DNS settings.  If you get an `ERR_CONNECTION_REFUSED` error, check your container's network configuration; you might need to manually configure DNS or adjust the network mode to ensure the container can access the external network.

### Step 3: Access and Manage

*   **Access Web UI:** Open `http://127.0.0.1:8000` in your browser.
*   **View Real-time Logs:** `docker-compose logs -f`
*   **Stop Container:** `docker-compose stop`
*   **Start Stopped Container:** `docker-compose start`
*   **Stop and Remove Container:** `docker-compose down`

## Web UI Feature Overview:

*   **Task Management:**
    *   **AI Task Creation:**  Describe your needs in natural language for automated task generation and AI analysis setup.
    *   **Visual Editing & Control:** Modify task parameters (keywords, prices, scheduling, etc.) and individually start/stop or delete tasks directly in the table.
    *   **Scheduled Execution:** Configure Cron expressions for automated, periodic task runs.
*   **Result Viewing:**
    *   **Card View:**  Displays matching items in a clear, visual card format.
    *   **Smart Filtering & Sorting:**  Filter for AI-recommended items and sort by crawl time, listing time, price, etc.
    *   **Detailed Views:** Click to see full item data and AI analysis results in JSON format.
*   **Running Logs:**
    *   **Real-time Log Stream:**  View detailed crawler logs in real-time on the web, to track progress and troubleshoot issues.
    *   **Log Management:** Supports auto-refresh, manual refresh, and one-click log clearing.
*   **System Settings:**
    *   **Status Check:**  Check the `.env` configuration, login status, and other critical dependencies.
    *   **Prompt Editing:**  Edit and save the `prompt` files used by the AI analysis directly on the web, to adjust the AI's logic in real-time.

## Workflow:

The diagram describes the core logic of a single monitoring task. The `web_server.py` acts as the main service and starts one or more of these task processes based on user actions or scheduled tasks.

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
    C -- No --> J[Paginate/Wait];
    J --> B;
    I --> C;
```

## Frequently Asked Questions (FAQ):

1.  **Q: "gbk' codec can't encode character" errors when running `login.py` or `spider_v2.py`?**
    *   **A:**  This is a common encoding issue on Windows. The code defaults to UTF-8.
    *   **Solution:** Set the `PYTHONUTF8=1` environment variable before running:

        ```bash
        set PYTHONUTF8=1
        python spider_v2.py
        ```

        Or use `chcp 65001` to switch the active code page to UTF-8.

2.  **Q:  `login.py` says `playwright install` is needed?**
    *   **A:** This means the required browser files for Playwright are missing. The recommended solution is to ensure all dependencies are installed correctly via `requirements.txt`:

        ```bash
        pip install -r requirements.txt
        ```

        If the problem persists, try manually installing the Chromium browser:

        ```bash
        playwright install chromium
        ```

3.  **Q:  "Request timed out" or "Connection error" when creating or running tasks?**
    *   **A:**  This is usually a network problem. Check:
        *   Your server's network connection.
        *   If in mainland China, you might need a proxy for accessing foreign AI services. Now configure `PROXY_URL` in your `.env` file.
        *   Verify the `OPENAI_BASE_URL` is correct and the service is running.

4.  **Q:  My AI model doesn't support image analysis?**
    *   **A:**  The core feature is multimodal analysis, so you **must** select an AI model with Vision/Multi-modal capabilities. Set `OPENAI_MODEL_NAME` in `.env` to a model that accepts image inputs (e.g., `gpt-4o`, `gemini-1.5-pro`, `deepseek-v2`, `qwen-vl-plus`).

5.  **Q:  Can I deploy on a Synology NAS via Docker?**
    *   **A:** Yes. The deployment is similar to standard Docker deployment:
        1.  Complete the `login.py` step on your computer (not the Synology) and generate `xianyu_state.json`.
        2.  Upload the entire project folder (including `.env` and `xianyu_state.json`) to a directory on your Synology.
        3.  In Container Manager (or Docker), run `docker-compose up -d` (via SSH or Task Scheduler).  Ensure the volume mapping in `docker-compose.yaml` points to the correct project folder on your Synology.

6.  **Q: How to configure Gemini / Qwen / Grok or other non-OpenAI LLMs?**
    *   ***A:** This project theoretically supports any model with an OpenAI-compatible API interface. Configure the `.env` variables correctly:
        *   `OPENAI_API_KEY`: Your model provider's API Key.
        *   `OPENAI_BASE_URL`: The API-Compatible Endpoint address. Check your model's documentation, typically in the format of `https://api.your-provider.com/v1` (note, no `/chat/completions` at the end).
        *   `OPENAI_MODEL_NAME`: The specific model name you are using, which needs to support image recognition, such as `gemini-2.5-flash`.
    *   **Example:** If your provider's documentation states the Completions interface as `https://xx.xx.com/v1/chat/completions`, then `OPENAI_BASE_URL` should be `https://xx.xx.com/v1`.

7.  **Q: Flagged by Xianyu after running for a while?**
    *   ***A:** This is an anti-scraping measure. Reduce detection risks:
        *   **Disable headless mode:** Set `RUN_HEADLESS=false` in `.env`. This runs the browser with a visible interface.  You can manually complete CAPTCHAs.
        *   **Reduce monitoring frequency:** Avoid running many tasks concurrently.
        *   **Use a clean network environment:** Frequent scraping may result in IP blocking.

8.  **Q: pyzbar installation fails on Windows?**
    *   **A:** pyzbar needs the zbar DLL on Windows.
    *   **Solutions (Windows):**
        *   **Method 1 (Recommended):** Use Chocolatey:

            ```cmd
            choco install zbar
            ```

        *   **Method 2:** Download `libzbar-64.dll` from [zbar releases](https://github.com/NaturalHistoryMuseum/pyzbar/releases), and put it in Python installation directory or add it to the system PATH.
        *   **Method 3:** Use conda:

            ```cmd
            conda install -c conda-forge zbar
            ```

    *   **Linux Users:** Install the system package:

        ```bash
        # Ubuntu/Debian
        sudo apt-get install libzbar0

        # CentOS/RHEL
        sudo yum install zbar

        # Arch Linux
        sudo pacman -S zbar
        ```

9.  **Q: `ModuleNotFoundError: No module named 'PIL'` when running `login.py`?**
    *   **A:** Usually caused by a low Python version or incomplete dependencies. This project recommends Python 3.10+.
    *   **Solutions:**
        *   Make sure you're using Python 3.10+.
        *   Reinstall dependencies:

            ```bash
            pip install -r requirements.txt
            ```

        *   If the problem persists, try installing Pillow:

            ```bash
            pip install Pillow
            ```

10. **Q:  404 errors when configuring AI API?**
    *   **A:** If you get 404 errors with the AI API, debug with the Alibaba Cloud API first, to make sure the basic functions are working, then try other AI API providers, because some API providers may have compatibility issues or need special configurations. Please check:
        *   Ensure that the address `OPENAI_BASE_URL` is filled in correctly and that the service is operating normally.
        *   Check whether the network connection is normal.
        *   Confirm that the API Key is correct and has access rights.
        *   Some API providers may require special request headers or parameter configuration, please refer to their official documentation.

## Acknowledgements:

This project is inspired by and built upon the following projects:

*   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

And also, thanks for the scripts from LinuxDo users:

*   [@jooooody](https://linux.do/u/jooooody/summary)

Thanks to Aider and Gemini for helping to free my hands and make the code writing feel like flying.

## Support & Sponsoring

If this project helps you, please consider buying a coffee for me, thank you very much for your support!

<table>
  <tr>
    <td><img src="static/zfb_support.jpg" width="200" alt="Alipay" /></td>
    <td><img src="static/wx_support.png" width="200" alt="WeChat Pay" /></td>
  </tr>
</table>

## ⚠️ Important Notes:

*   Please adhere to Xianyu's user agreement and `robots.txt` to avoid server load and account restrictions.
*   This project is for educational and research purposes only; do not use it for illegal activities.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)