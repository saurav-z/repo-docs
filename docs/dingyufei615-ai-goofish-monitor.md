# AI-Powered Xianyu (Goofish) Monitor: Real-time Item Tracking and Smart Analysis

**Tired of missing out on great deals?** This project is a powerful, AI-driven tool that monitors Xianyu (Goofish) for you, leveraging AI to analyze listings and deliver real-time notifications for items that match your criteria.  [View the original repository](https://github.com/dingyufei615/ai-goofish-monitor)

## Key Features:

*   🔍 **AI-Driven Task Creation:** Describe your desired item in natural language and let AI build your monitoring task.
*   🖥️ **Intuitive Web UI:**  Manage tasks, view real-time logs, and filter results, all through a user-friendly web interface.
*   ⚙️ **Multi-Task Concurrency:** Monitor multiple keywords simultaneously without interference.
*   ⚡ **Real-time Processing:** Instant analysis and notification upon discovery of new listings.
*   🧠 **Deep AI Analysis:**  Uses multimodal large language models (like GPT-4o) for comprehensive image and text analysis.
*   ⚙️ **Highly Customizable:** Configure keywords, price ranges, AI prompts, and notification settings for each task.
*   🔔 **Instant Notifications:** Receive alerts via ntfy.sh, WeChat group bots, and Bark.
*   📅 **Scheduled Task Execution:** Utilize Cron expressions for automated, periodic monitoring.
*   🐳 **Dockerized Deployment:**  Deploy quickly and consistently with provided Docker Compose configuration.
*   🛡️ **Robust Anti-Scraping Measures:**  Mimics human behavior with random delays and user actions to enhance stability.

##  Get Started Quickly!

The Web UI provides the best user experience.

### 1. Prerequisites

*   **Python:**  Python 3.10 or later is recommended.
*   Clone the repository:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

*   Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuration

1.  **Environment Variables:**
    *   Create a `.env` file (copy `.env.example` and rename).
    *   Configure the following in your `.env` file:

        | Variable             | Description                                   | Required | Notes                                                                                                          |
        | -------------------- | --------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
        | `OPENAI_API_KEY`     | Your AI model provider API key                | Yes      |                                                                                                                |
        | `OPENAI_BASE_URL`    | API endpoint (OpenAI format compatible)       | Yes      |                                                                                                                |
        | `OPENAI_MODEL_NAME`  | Your AI model name                           | Yes      | Must be a multimodal model (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`)                              |
        | `PROXY_URL`          | (Optional) HTTP/S proxy                       | No       | `http://` and `socks5://` supported                                                                          |
        | `NTFY_TOPIC_URL`     | (Optional) ntfy.sh topic URL                | No       |                                                                                                                |
        | ... (See Original README for other optional variables like GOTIFY, BARK, WX_BOT, etc.) | ... | ... | ...|
        | `WEB_USERNAME`       | Web UI login username                | No |  Default: `admin`.  **Change in production.** |
        | `WEB_PASSWORD`        | Web UI login password               | No  | Default: `admin123`.  **Change in production.** |

    *   **Debugging Tip:** If you encounter 404 errors, test with a provider like Ali or Volcano to ensure basic functionality before trying other APIs.

    *   **Security Note:** The Web UI uses Basic Auth.  **CHANGE THE DEFAULT `admin`/`admin123` CREDENTIALS IN PRODUCTION!**

2.  **Obtaining Login Credentials (Crucial!)**
    *   **Recommended:  Via Web UI:**
        1.  Start the web server (Step 3).
        2.  Go to "System Settings" in the Web UI.
        3.  Click "Manual Update" under "Login State File."
        4.  Follow the instructions to install the browser extension, log in to Xianyu, and extract/paste the login state.
    *   **Alternative:  Login Script (if you can run a browser locally):**
        ```bash
        python login.py
        ```
        This will open a browser for you to log in via QR code.  It then generates `xianyu_state.json`.

### 3. Launch the Web Server

```bash
python web_server.py
```

### 4. Start Monitoring!

1.  Open the Web UI:  `http://127.0.0.1:8000`
2.  In "Task Management," click "Create New Task."
3.  Describe your desired item (e.g., "Sony A7M4 camera, like new, under 13,000 RMB, shutter count below 5000").
4.  Click "Create" to let AI generate the monitoring criteria.
5.  Start the task manually or schedule it.

### 🐳 Docker Deployment (Recommended)

Docker provides a streamlined deployment solution.

#### 1. Prepare (Similar to Local)

1.  Install Docker and Docker Compose.
2.  Clone the repository.
3.  Create and configure the `.env` file.
4.  **Get Login Credentials:** Use the Web UI (after launching the container):
    1.  Run `docker-compose up -d`.
    2.  Access the Web UI at `http://127.0.0.1:8000`.
    3.  Go to "System Settings," then "Manual Update" to follow the login steps.

#### 2. Run the Docker Container

```bash
docker-compose up --build -d
```

#### 3. Access & Manage

*   Web UI: `http://127.0.0.1:8000`
*   View Logs: `docker-compose logs -f`
*   Stop: `docker-compose stop`
*   Start: `docker-compose start`
*   Stop and Remove: `docker-compose down`

## Web UI Features:

*   **Task Management:** AI task creation, visual editing, scheduling (Cron), start/stop/delete.
*   **Results View:** Card-based item display, AI-powered filtering, sorting, deep item details.
*   **Real-time Logs:** Live logs, auto-refresh, and log management.
*   **System Settings:** Status checks, AI prompt editing.

##  Workflow Overview

```mermaid
graph TD
    A[Start Monitoring Task] --> B[Task: Search for Items];
    B --> C{New Item Found?};
    C -- Yes --> D[Get Item Details & Seller Info];
    D --> E[Download Item Images];
    E --> F[Call AI Analysis];
    F --> G{AI Recommends?};
    G -- Yes --> H[Send Notification];
    H --> I[Save to JSONL];
    G -- No --> I;
    C -- No --> J[Pagination/Wait];
    J --> B;
    I --> C;
```

##  Authentication

The Web UI is protected by Basic Authentication.  Refer to the `AUTH_README.md` file for detailed information.

##  Frequently Asked Questions (FAQ)

Get answers to common questions in the [FAQ.md](FAQ.md) document.

##  Acknowledgments

Special thanks to the projects and communities listed in the original README.

##  Support & Sponsoring

Consider supporting the project (details in the original README).

## ⚠️ Important Notes

*   Comply with Xianyu's terms of service and robots.txt.
*   For research/personal use only; no illegal activities.
*   Released under the MIT License (see LICENSE).
*   Project contributors are not liable for any damages.
*   Refer to [DISCLAIMER.md](DISCLAIMER.md) for more information.