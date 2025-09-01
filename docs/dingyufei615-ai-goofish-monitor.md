# AI-Powered Goofish (闲鱼) Smart Monitor: Your AI Assistant for Finding Deals

**Automatically find the best deals on Xianyu (Goofish) with AI-driven analysis and a user-friendly web interface.**

[View Original Repository](https://github.com/dingyufei615/ai-goofish-monitor)

## Key Features

*   ✅ **Intuitive Web UI:** Manage tasks, edit AI criteria, view logs, and filter results all within a web interface.
*   🤖 **AI-Driven Task Creation:** Describe your desired item in natural language, and the AI creates a monitoring task with complex filtering.
*   ⚙️ **Multi-Task Concurrency:** Monitor multiple keywords simultaneously, with each task running independently.
*   ⚡️ **Real-time Analysis:** Analyze new listings instantly, avoiding batch processing delays.
*   🧠 **Deep AI Analysis:** Leverage multimodal large language models (like GPT-4o) to analyze item descriptions, images, and seller profiles for accurate filtering.
*   🛠️ **Highly Customizable:** Configure each task with unique keywords, price ranges, filtering conditions, and AI analysis prompts.
*   🔔 **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat Work group bots, and [Bark](https://bark.day.app/) when matching items are found.
*   📅 **Scheduled Tasks:** Utilize Cron expressions for automated, time-based task execution.
*   🐳 **One-Click Docker Deployment:** Simplify deployment with pre-configured `docker-compose` files.
*   🛡️ **Robust Anti-Scraping:** Employ realistic user behavior, including random delays, to improve stability.

## Web UI Screenshots

*   **Task Management:**
    ![img.png](static/img.png)
*   **Monitoring:**
    ![img_1.png](static/img_1.png)
*   **Notification Example:**
    ![img_2.png](static/img_2.png)

## Getting Started

This guide provides instructions for both local deployment and Docker deployment.

### Local Setup (Recommended for initial setup and debugging)

1.  **Prerequisites:**

    *   **Python 3.10+:** Ensure you have Python 3.10 or higher installed. Older versions may cause dependency issues (e.g., `PIL` errors).
2.  **Clone and Install:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables (.env):**

    *   Copy the example file:

        ```bash
        # Windows
        copy .env.example .env
        # Linux/macOS
        cp .env.example .env
        ```

    *   Edit the `.env` file with your API keys, model names, and notification settings.  Key variables include:

        *   `OPENAI_API_KEY`:  Your AI model API key.
        *   `OPENAI_BASE_URL`: The API endpoint for your AI model.
        *   `OPENAI_MODEL_NAME`: The specific multimodal model you are using (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`).
        *   `PROXY_URL`: (Optional) Proxy settings for network access.
        *   `NTFY_TOPIC_URL`, `GOTIFY_URL/TOKEN`, `BARK_URL`, `WX_BOT_URL`, `WEBHOOK_URL`: (Optional) Configure for desired notification services.
        *   `WEB_USERNAME`, `WEB_PASSWORD`:  Web UI login credentials.  **Change these from the defaults in a production environment!**
        *   `LOGIN_IS_EDGE`, `RUN_HEADLESS`, `AI_DEBUG_MODE`, `SKIP_AI_ANALYSIS`:  Useful flags for tailoring crawler behavior.
    *   **Debugging Note:** If you encounter 404 errors when setting up your AI API, test with APIs like Alibaba Cloud or Volcano Engine to confirm basic functionality before trying others.

    >  **Important Security Note:** The Web UI utilizes Basic Authentication. The default username/password is `admin`/`admin123`. **Change these immediately in production!**
4.  **Obtain Login State (Crucial!):**  You must provide valid login credentials to allow the crawler to access Xianyu.
    *   **Recommended: Web UI Method:**
        1.  **Start the Web Server (Step 3).**
        2.  Go to "System Settings" in the Web UI.
        3.  Click "Manual Update" under "Login State File".
        4.  Follow the on-screen instructions:
            *   Install the [Xianyu login state extension](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) in Chrome.
            *   Log in to Xianyu through the extension.
            *   Click the extension icon and "Extract Login State".
            *   Copy the output and paste it into the Web UI.
    *   **Alternative: Login Script (For local/desktop environments):**
        ```bash
        python login.py
        ```
        A browser window will open. Scan the QR code with your Xianyu app to log in. A `xianyu_state.json` file will be created.
5.  **Start the Web Server:**

    ```bash
    python web_server.py
    ```

6.  **Use the Web UI:**  Access the Web UI at `http://127.0.0.1:8000`.
    1.  Create a new task in "Task Management."
    2.  Describe your desired item in natural language (e.g., "Looking for a Sony a7M4 camera, excellent condition, budget under 13000").
    3.  The AI generates the task configuration.
    4.  Start your task and start monitoring!

### Docker Deployment (Highly Recommended)

1.  **Prerequisites:**

    *   **Docker:** Install [Docker Engine](https://docs.docker.com/engine/install/).
    *   Follow the instructions from the **Local Setup**, clone the repository, and create the `.env` file.
    *   Login to Goofish and setup the Login State by using the Web UI (Step 2 in the Local Setup instructions, and the last section of the Docker Setup instructions).

2.  **Run the Docker Container:**

    ```bash
    docker-compose up --build -d
    ```

    This builds and runs the containers in the background.
3.  **Login & Configure Login State**: After your container starts, you MUST set your Xianyu login state through the Web UI.
    1.  Open the Web UI at `http://127.0.0.1:8000`.
    2.  Follow the Web UI login process as described in Step 2 of the Local Setup.

4.  **Access and Manage:**

    *   **Web UI:** `http://127.0.0.1:8000`
    *   **Logs:** `docker-compose logs -f`
    *   **Stop:** `docker-compose stop`
    *   **Start:** `docker-compose start`
    *   **Remove:** `docker-compose down`

## Web UI Features

*   **Task Management:** Create tasks using natural language prompts and modify them through a web interface.  Schedule tasks with Cron expressions.
*   **Result Viewing:** Browse results with a card-based interface, AI recommendations, and sort and filter options.  View detailed item data and AI analysis results.
*   **Real-time Logs:** Monitor crawler activity in real-time within the UI.
*   **System Settings:** Test configurations, edit AI prompts, and manage authentication.

## Workflow

[Insert Workflow Diagram Here - see the original README for the mermaid code.  You can generate the image using a tool like mermaid-js.org ]

## Web UI Authentication

[See the Original README for authentication information.]

## Frequently Asked Questions (FAQ)

[Link to FAQ.md - e.g.,  👉  **[FAQ.md](FAQ.md)**]

## Acknowledgements

[See the Original README for Acknowledgements.]

## Disclaimer

*   Comply with Xianyu's terms of service and robots.txt.
*   For research and educational use only.
*   No warranty; use at your own risk.
*   See the [DISCLAIMER.md](DISCLAIMER.md) file.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)