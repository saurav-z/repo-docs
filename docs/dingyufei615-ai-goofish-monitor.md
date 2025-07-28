# AI-Powered Xianyu (Goofish) Monitor: Effortlessly Track & Analyze Secondhand Goods

This project leverages AI to monitor and analyze secondhand goods listings on Xianyu (Goofish) using Playwright. Create, monitor, and manage your tasks through a user-friendly web interface or command-line tools; [view the original repository](https://github.com/dingyufei615/ai-goofish-monitor).

## Key Features:

*   **AI-Driven Task Creation:** Describe your desired item in natural language to generate complex monitoring tasks with AI-powered filtering.
*   **Visual Web Interface:** Manage tasks, view real-time logs, filter results, and edit AI criteria directly from the web UI.
*   **Multi-Task Concurrency:** Monitor multiple keywords simultaneously, with independent task execution.
*   **Real-time Stream Processing:** Analyze new listings instantly, eliminating batch processing delays.
*   **Deep AI Analysis:** Utilize multimodal large language models (e.g., GPT-4o) to analyze listings based on image and seller data, for accurate filtering.
*   **Highly Customizable:** Configure individual tasks with specific keywords, price ranges, filtering conditions, and AI analysis prompts.
*   **Instant Notifications:** Receive notifications of AI-recommended items via ntfy.sh to your phone or desktop.
*   **Robust Anti-Scraping:** Mimics human behavior with random delays and user interactions to enhance stability.

## Quickstart: Web UI Deployment (Recommended)

The Web UI offers the best user experience for interacting with the project.

### Step 1: Environment Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Configuration

1.  **Configure Environment Variables:** Create a `.env` file by copying the example:

    *   **Windows:**

        ```cmd
        copy .env.example .env
        ```

    *   **Linux/macOS:**

        ```bash
        cp .env.example .env
        ```

    Edit the `.env` file with your API keys, notification settings, and other configurations.

2.  **Get Login Status (Important!):**  Run the login script to generate the `xianyu_state.json` file, which is essential for the crawler to access Xianyu in a logged-in state.

    ```bash
    python login.py
    ```

    A browser window will open. **Use your Xianyu App to scan the QR code** to log in. Upon successful login, the script will close and create `xianyu_state.json` in the project root.

### Step 3: Start the Web Server

1.  **Run the Web Server:**

    ```bash
    python web_server.py
    ```

### Step 4: Start Monitoring

1.  **Access the Web UI:** Open `http://127.0.0.1:8000` in your browser.
2.  **Create a New Task:** In the "Task Management" page, click "Create New Task."
3.  **Describe Your Needs:** Use natural language (e.g., "I want to buy a Sony A7M4 camera, mint condition, budget under 13000, shutter count less than 5000") and provide the task name and keywords.
4.  **Let AI Generate:** Click "Create." The AI will automatically generate sophisticated analysis criteria.
5.  **Start Monitoring:** Return to the main interface and click "🚀 Start All" to activate your monitoring tasks.

## Docker Deployment (Recommended)

Docker simplifies deployment by packaging the application and all dependencies into a standardized unit.

### Step 1: Environment Setup (Similar to Local Deployment)

1.  **Install Docker:** Ensure Docker Engine is installed.  ([Docker Engine](https://docs.docker.com/engine/install/))

2.  **Clone the project and configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env` file:**  Follow the instructions in the [Quickstart](#quickstart-web-ui-recommended) section to create and populate your `.env` file in the project root.

4.  **Get Login Status (Crucial!):**  **Run the login script on your host machine (outside Docker)** to generate the `xianyu_state.json` file. Login requires user interaction (QR code scan) and cannot be done during Docker build.

    ```bash
    # Ensure you have Python and dependencies installed locally
    pip install -r requirements.txt
    python login.py
    ```

    After successful login (QR code scan), `xianyu_state.json` will be created in the project root.

### Step 2: Run the Docker Container

1.  **Start Container:** Use the `docker-compose` command:

    ```bash
    docker-compose up -d
    ```

    This starts the service in the background. `docker-compose` uses `.env` and `docker-compose.yaml` to build and launch the container.

    Troubleshoot network issues inside the container or use a proxy if necessary.

### Step 3: Access and Manage

*   **Web UI Access:**  Open `http://127.0.0.1:8000` in your browser.
*   **Real-time Logs:**  `docker-compose logs -f`
*   **Stop Container:** `docker-compose stop`
*   **Start Stopped Container:** `docker-compose start`
*   **Stop and Remove Container:** `docker-compose down`

## Web UI Feature Overview

*   **Task Management:**
    *   **AI Task Creation:** Create monitoring tasks with AI-generated analysis rules using natural language descriptions.
    *   **Visual Editing:** Modify task parameters like keywords and price ranges directly in a table.
    *   **Start/Stop Control:** Enable/disable individual tasks or all tasks at once.
*   **Result Viewing:**
    *   **Card View:** Display matched items in an image and text card format.
    *   **Smart Filtering:** Filter items based on AI "recommended" tags.
    *   **Detailed View:** See the full data and AI analysis results in a JSON format.
*   **Running Logs:**
    *   **Real-time Log Stream:** Monitor detailed logs as they run, to track progress and troubleshoot.
*   **System Settings:**
    *   **Status Check:** Verify `.env` configurations, login status, and dependencies.
    *   **Prompt Editing:** Edit and save `prompt` files to adjust AI logic.

## Advanced Command-Line Usage

For those preferring command-line control, the project supports independent script execution.

### Start Monitoring

Run the main crawler script to load and execute all enabled tasks from `config.json`:

```bash
python spider_v2.py
```

**Debugging Mode:**  Limit the number of items processed per task using the `--debug-limit` argument.

```bash
# Process only the first 2 new items per task
python spider_v2.py --debug-limit 2
```

### Create New Tasks with Script

The `prompt_generator.py` script streamlines new task creation from the command line:

```bash
python prompt_generator.py \
  --description "I want to buy a Sony A7M4 camera, mint condition, budget under 13000, shutter count less than 5000. Must be a China-market version with all original accessories. Preferably a private seller, no businesses or resellers." \
  --output prompts/sony_a7m4_criteria.txt \
  --task-name "Sony A7M4" \
  --keyword "a7m4" \
  --min-price "10000" \
  --max-price "13000"
```

This creates a new `_criteria.txt` file and adds the task to `config.json`.

## Workflow

```mermaid
graph TD
    A[Start Main Program] --> B{Read config.json};
    B --> C{Start Multiple Monitoring Tasks Concurrently};
    C --> D[Task: Search Items];
    D --> E{New Item Found?};
    E -- Yes --> F[Fetch Item Details & Seller Information];
    F --> G[Download Item Images];
    G --> H[Call AI for Analysis];
    H --> I{AI Recommends?};
    I -- Yes --> J[Send ntfy Notification];
    J --> K[Save Record to JSONL];
    I -- No --> K;
    E -- No --> L[Paging/Wait];
    L --> D;
    K --> E;
```

## Technology Stack:

*   **Core Framework:** Playwright (Asynchronous) + asyncio
*   **Web Service:** FastAPI
*   **AI Model:** OpenAI API (Supports GPT-4o and other multimodal models)
*   **Notification Service:** ntfy
*   **Configuration:** JSON
*   **Dependency Management:** pip

## Project Structure

```
.
├── .env                # Environment variables (API keys)
├── .gitignore          # Git ignore configuration
├── config.json         # Configuration file for all monitoring tasks
├── login.py            # Run to obtain and save login cookies
├── spider_v2.py        # Core crawler program
├── prompt_generator.py # AI analysis criteria generation script
├── web_server.py       # Web server main program
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── prompts/            # AI analysis prompts
│   ├── base_prompt.txt
│   └── ..._criteria.txt
├── static/             # Web frontend static files
│   ├── css/style.css
│   └── js/main.js
├── templates/          # Web frontend templates
│   └── index.html
├── images/             # Downloaded item images (auto-created)
├── logs/               # Running logs (auto-created)
└── jsonl/              # Results from each task (auto-created)
```

## Frequently Asked Questions (FAQ)

1.  **Q: Getting `gbk' codec can't encode character` errors when running `login.py` or `spider_v2.py`?**
    *   **A:** This is a common encoding issue on Windows. The code and logs default to UTF-8.
    *   **Solution:** Set the `PYTHONUTF8` environment variable before running:

        ```bash
        set PYTHONUTF8=1
        python spider_v2.py
        ```

        Or, use `chcp 65001` to set the active code page to UTF-8.

2.  **Q: Getting an error saying `playwright install` is needed when running `login.py`?**
    *   **A:** This means the required browser files for Playwright are missing. The recommended solution is to ensure all dependencies are correctly installed via `requirements.txt`. Run:

        ```bash
        pip install -r requirements.txt
        ```

        If the issue persists, try installing Chromium manually:

        ```bash
        playwright install chromium
        ```

3.  **Q: Getting "Request timed out" or "Connection error" during task creation or runtime?**
    *   **A:**  This likely indicates a network issue.  Verify the following:
        *   Your server's network connectivity.
        *   If in mainland China, you may need to set up a proxy for AI services (OpenAI, Gemini). Configure the `PROXY_URL` variable in `.env`.
        *   Confirm that the `OPENAI_BASE_URL` is correctly filled, and the AI service is working.

4.  **Q: My AI model doesn't support image analysis, what should I do?**
    *   **A:**  The core of the project relies on multimodal analysis.  **You must** select an AI model that supports image recognition (Vision / Multi-modal). Replace `OPENAI_MODEL_NAME` in your `.env` file with a suitable model (e.g., `gpt-4o`, `gemini-1.5-pro`, `deepseek-v2`, `qwen-vl-plus`).

5.  **Q: Can I deploy on a Synology NAS via Docker?**
    *   **A:** Yes. The steps are similar to standard Docker deployment:
        1.  Complete the `login.py` step on your local computer (not on the NAS) to generate `xianyu_state.json`.
        2.  Upload the entire project folder (including `.env` and `xianyu_state.json`) to a directory on your Synology NAS.
        3.  In Container Manager (or the older Docker), use `docker-compose up -d` (via SSH or Task Scheduler) to start the project. Ensure that the volume mapping in `docker-compose.yaml` correctly points to your project folder on the NAS.

6.  **Q: How to configure and use Gemini / Qwen / Grok or other non-OpenAI LLMs?**
    *   **A:** This project supports any model with an OpenAI-compatible API interface. Configure the following variables in your `.env` file:
        *   `OPENAI_API_KEY`:  Your API key from the model provider.
        *   `OPENAI_BASE_URL`:  The API-compatible endpoint URL from your model provider. Check your provider's documentation for the correct format, usually `https://api.your-provider.com/v1` (without `/chat/completions` at the end).
        *   `OPENAI_MODEL_NAME`: The specific model name you're using, which must support image recognition, e.g., `gemini-2.5-flash`.
        *   **Example:** If the Completions endpoint from your provider is `https://xx.xx.com/v1/chat/completions`, then set `OPENAI_BASE_URL` to `https://xx.xx.com/v1`.

7.  **Q:  Getting flagged for "abnormal traffic" or CAPTCHAs after running for a while?**
    *   **A:**  This is part of Xianyu's anti-scraping measures.  Try the following:
        *   **Disable Headless Mode:**  Set `RUN_HEADLESS=false` in `.env`.  The browser will run with a visible window, allowing you to manually solve CAPTCHAs.
        *   **Reduce Monitoring Frequency:**  Avoid running too many monitoring tasks simultaneously.
        *   **Use a Clean Network Environment:**  Frequent scraping may lead to IP address flagging.

## Acknowledgements

This project is built upon the following open-source projects, and I am grateful for their contributions:

*   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

Thanks to friends at LinuxDo for script contributions

*   [@jooooody](https://linux.do/u/jooooody/summary)

Thanks to Aider and Gemini for liberating my hands, and the code is like flying.

## ⚠️ Important Notes

*   Respect Xianyu's terms of service and the robots.txt rules. Avoid excessive requests to prevent server overload or account restrictions.
*   This project is for educational and research purposes only. Do not use it for illegal activities.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)