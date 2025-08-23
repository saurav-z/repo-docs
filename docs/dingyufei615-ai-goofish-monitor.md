# AI-Powered Goofish Monitor: Effortlessly Track and Analyze Xianyu (Goofish) Listings

**Tired of missing out on great deals?** This project is your all-in-one solution for real-time monitoring and intelligent analysis of Xianyu (Goofish) listings, powered by AI and featuring a user-friendly web interface. [Visit the original repository](https://github.com/dingyufei615/ai-goofish-monitor) for the complete source code and documentation.

## Key Features:

*   ✅ **Web UI for Easy Management:** A fully-featured web interface for task management, AI prompt editing, real-time log viewing, and result filtering, eliminating the need for command-line interaction.
*   💬 **AI-Driven Task Creation:** Simply describe your desired purchase in natural language, and the system will automatically create a monitoring task with complex filtering logic.
*   ⚙️ **Concurrent Multi-Tasking:** Monitor multiple keywords simultaneously via `config.json`, with each task running independently.
*   🚀 **Real-Time Processing:** Analyze new listings instantly, avoiding batch processing delays.
*   🧠 **Deep AI Analysis:** Integrates multimodal large language models (like GPT-4o) to analyze product descriptions, images, and seller profiles for accurate filtering.
*   🛠️ **Highly Customizable:** Each monitoring task can be configured with unique keywords, price ranges, filtering criteria, and AI analysis prompts.
*   🔔 **Instant Notifications:** Receive immediate alerts via [ntfy.sh](https://ntfy.sh/), Enterprise WeChat group bots, and [Bark](https://bark.day.app/) to your phone or desktop.
*   📅 **Scheduled Task Execution:** Supports cron expressions for setting independent schedules for each task.
*   🐳 **Dockerized Deployment:** Provides a `docker-compose` configuration for quick and standardized containerized deployment.
*   🛡️ **Robust Anti-Scraping Strategies:** Employs human-like interactions, including random delays and user behavior simulation, to enhance stability.

## Core Functionality in a Nutshell

The system efficiently monitors Xianyu listings, utilizing AI for intelligent filtering, and delivering timely notifications.

## Getting Started: A Step-by-Step Guide

### 1. Prerequisites

*   **Python 3.10+:** Essential for dependencies and smooth execution.
*   Clone the repository:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

*   Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuration:

1.  **Environment Variables:**

    *   Create a `.env` file by copying `.env.example`.
    *   Edit the `.env` file with your API keys and preferences.

    | Variable              | Description                                              | Required | Notes                                                                                                       |
    | :-------------------- | :------------------------------------------------------- | :------- | :---------------------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`      | Your AI model provider's API Key.                        | Yes      |                                                                                                             |
    | `OPENAI_BASE_URL`     | AI model API endpoint (OpenAI-compatible).               | Yes      | e.g., `https://ark.cn-beijing.volces.com/api/v3/`                                                      |
    | `OPENAI_MODEL_NAME`   | The specific multimodal model to use (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`). | Yes      | **Must** select a multimodal model that supports image analysis.                                           |
    | `PROXY_URL`           | (Optional) HTTP/S proxy for bypassing network restrictions. | No       | Supports `http://` and `socks5://` formats.                                                              |
    | `NTFY_TOPIC_URL`      | (Optional) ntfy.sh topic URL for notifications.         | No       | Leave blank to disable ntfy notifications.                                                              |
    | ...                   | ... (Refer to original README for more options)          |          | ...                                                                                                         |

2.  **Get Login Credentials (Crucial!)**

    *   **Recommended: Update via Web UI**
        1.  Start the web server (Step 3).
        2.  Go to "System Settings" in the Web UI.
        3.  Click "Manually Update" in the "Login Status File" section.
        4.  Follow on-screen instructions to use the [Xianyu Login State Extractor](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) browser extension.

    *   **Alternative: Run Login Script (if you have a GUI environment)**
        ```bash
        python login.py
        ```
        Log in via the mobile Xianyu app by scanning the QR code.

### 3. Start the Web Server

```bash
python web_server.py
```

### 4. Start Monitoring!

1.  Open the Web UI in your browser: `http://127.0.0.1:8000`.
2.  Go to "Task Management" and create a new task.
3.  Describe your purchase needs in natural language.
4.  The AI automatically configures the task.
5.  Start the task!

## 🐳 Docker Deployment (Recommended)

Streamline deployment with Docker for portability and consistency.

### 1. Docker Setup:

1.  **Install Docker Engine**
2.  Clone the project & configure `.env` as above.
3.  Get Login Credentials (Use Web UI after starting the container – see step 2 in the Deployment Section)

### 2. Run the Docker Container

```bash
docker-compose up --build -d
```

### 3. Access and Manage

*   **Web UI:** `http://127.0.0.1:8000`
*   **Logs:** `docker-compose logs -f`
*   **Stop:** `docker-compose stop`
*   **Start:** `docker-compose start`
*   **Remove:** `docker-compose down`

## Additional Information:

*   **Web UI Overview:** Task management, result browsing, real-time logs, and system settings.
*   **Workflow:** A diagram illustrating the task lifecycle.
*   **Authentication:** Basic authentication is used for web UI security (Username/Password are configurable in `.env`.)
*   **FAQ:** Comprehensive troubleshooting at [FAQ.md](FAQ.md)
*   **Acknowledgements & Support:**  See the original README for details.

## Important Notes:

*   Respect Xianyu's terms of service and robots.txt.
*   This project is for educational and research purposes only.
*   Refer to the [DISCLAIMER.md](DISCLAIMER.md) file for more details.