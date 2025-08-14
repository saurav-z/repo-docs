# AI-Powered Goofish (闲鱼) Monitor: Smart Shopping Made Easy

**Tired of missing out on great deals?** This intelligent monitor uses AI to analyze and filter Goofish (闲鱼) listings, sending you instant notifications for the items you want, all within an intuitive web interface.  [View the original repo](https://github.com/dingyufei615/ai-goofish-monitor).

## Key Features

*   ✅ **Web-Based Management:**  Effortlessly manage tasks, edit AI criteria, and view real-time logs through a user-friendly web UI.
*   🤖 **AI-Driven Task Creation:**  Simply describe your desired item in natural language, and the AI will create a custom monitoring task with complex filtering.
*   ⚙️ **Concurrent Multi-Tasking:** Monitor multiple keywords simultaneously with independent, non-interfering tasks managed via `config.json`.
*   ⚡ **Real-Time Analysis:**  Receive immediate analysis of new listings, eliminating batch processing delays.
*   🧠 **Deep AI Insights:** Leverages multimodal large language models (e.g., GPT-4o) to deeply analyze product descriptions, images, and seller profiles for precise filtering.
*   🛠️ **Highly Customizable:** Tailor each monitoring task with unique keywords, price ranges, filtering rules, and AI prompts.
*   🔔 **Instant Notifications:** Get notified instantly via [ntfy.sh](https://ntfy.sh/), WeChat group bots, or [Bark](https://bark.day.app/) on your phone or desktop.
*   🗓️ **Scheduled Task Execution:**  Utilize Cron expressions to schedule each task for automatic, periodic monitoring.
*   🐳 **Docker Deployment:** Deploy quickly and easily with a pre-configured `docker-compose` setup.
*   🛡️ **Robust Anti-Scraping:** Mimics human behavior with random delays and user actions to improve stability and bypass anti-bot measures.

## Screenshots

**Task Management Dashboard**
![img.png](static/img.png)

**Monitoring Dashboard**
![img_1.png](static/img_1.png)

**Notification Example**
![img_2.png](static/img_2.png)

## Getting Started

### 1. Prerequisites

> ⚠️ **Python Version:**  Python 3.10 or higher is recommended for local development and debugging. Lower versions may cause dependency installation issues.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuration

1.  **Set Environment Variables:** Copy `.env.example` to `.env` and customize the settings.

    *   **Windows:**

        ```cmd
        copy .env.example .env
        ```

    *   **Linux/macOS:**

        ```shell
        cp .env.example .env
        ```

    **Environment Variables:**

    | Variable            | Description                                          | Required? | Notes                                                                                                                                   |
    | :------------------ | :--------------------------------------------------- | :-------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`    | Your AI model provider's API Key.                     | Yes       |                                                                                                                                       |
    | `OPENAI_BASE_URL`   | Base URL for the AI model API (OpenAI-compatible).  | Yes       |                                                                                                                                       |
    | `OPENAI_MODEL_NAME` | The specific AI model to use.                        | Yes       | **Must** be a multimodal model supporting image analysis (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`).                       |
    | `PROXY_URL`         | (Optional) HTTP/S proxy for network access.          | No        | Supports `http://` and `socks5://`.  Example: `http://127.0.0.1:7890`.                                                                 |
    | `NTFY_TOPIC_URL`    | (Optional) [ntfy.sh](https://ntfy.sh/) topic URL.     | No        |                                                                                                                                       |
    | ... (Other notification and configuration options) ... | ... (See the original README) ... | ... | ... |
    | `SERVER_PORT`       | Web UI Port.                                         | No        | Defaults to `8000`.                                                                                                                     |
    | `WEB_USERNAME`      | Web UI Username                                    | No        | Defaults to `admin`. **Change in production!**                                                                                       |
    | `WEB_PASSWORD`      | Web UI Password                                    | No        | Defaults to `admin123`. **Use a strong password in production!**                                                                       |

    > 💡 **Debugging:**  If you encounter 404 errors with your AI API, test with an API from a service like AliCloud or Volcano Engine to verify core functionality before troubleshooting.

    > 🔐 **Security:** The web interface uses Basic Authentication.  Default credentials are `admin` / `admin123`.  **Change these in production!**

2.  **Acquire Login State (Crucial!):** The scraper requires valid login credentials to access Goofish. We recommend using the Web UI:

    **Recommended Method: Web UI Update**

    1.  Skip this step and start the web server (Step 3).
    2.  Navigate to **"System Settings"** in the Web UI.
    3.  Click the **"Manual Update"** button under "Login State File."
    4.  Follow the on-screen instructions, which involve:
        *   Installing the [Goofish Login State Extractor](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) Chrome extension.
        *   Logging into Goofish in Chrome.
        *   Using the extension to extract your login state.
        *   Pasting the extracted state into the Web UI.

    **Alternative: Run Login Script (If you have a desktop environment)**

    ```bash
    python login.py
    ```

    This will open a browser window. Scan the QR code with your Goofish app to log in.  A `xianyu_state.json` file will be created in the project root.

### 3. Start the Web Server

```bash
python web_server.py
```

### 4. Start Monitoring!

1.  Open your browser to `http://127.0.0.1:8000`.
2.  Go to **"Task Management"** and click **"Create New Task."**
3.  Describe your desired purchase in natural language (e.g., "I want a used Sony A7M4 camera, 95% new or better, budget under $1500, shutter count less than 5000").
4.  Click "Create" – the AI will generate the monitoring criteria.
5.  Start the task manually or configure a schedule.

## Docker Deployment (Recommended)

Docker streamlines deployment.

### 1. Prepare (similar to local setup)

1.  **Install Docker:** Make sure Docker Engine is installed.
2.  **Clone & Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```
3.  **Create `.env`:** Configure your `.env` file as described in "Getting Started."
4.  **Get Login State (Critical for Docker!):** Set the login state **after** starting the container:

    1.  Run `docker-compose up -d`.
    2.  Open `http://127.0.0.1:8000` in your browser.
    3.  Go to **"System Settings"** and click **"Manual Update."**
    4.  Follow the instructions as described in the Web UI login method.

### 2. Run the Docker Container

```bash
docker-compose up --build -d
```

This starts the service in detached mode.

**Important considerations for Docker:**
*   The project utilizes Python 3.11 inside the Dockerfile, so you don't have to worry about local Python version compatibility.
*   If you encounter network issues within the container, troubleshoot or configure proxy settings.

### 3. Access & Manage

-   **Web UI:**  `http://127.0.0.1:8000`
-   **View Logs:** `docker-compose logs -f`
-   **Stop Container:** `docker-compose stop`
-   **Start Stopped Container:** `docker-compose start`
-   **Stop and Remove Container:** `docker-compose down`

## Web UI Functionality at a Glance

*   **Task Management**: Create AI-driven tasks, visualize and edit task parameters, schedule automated runs.
*   **Result Viewing**:  Browse results with card-based layouts. Sort and filter recommendations by different criteria.
*   **Real-time Logging**: View detailed logs within the web interface, manage and troubleshoot issues easily.
*   **System Settings**: Check dependencies, edit AI prompts, and update login state.

## Workflow

[Insert Mermaid diagram as in original README]

## Authentication

### Authentication Configuration

Web UI has Basic Authentication enabled, ensuring authorized access only.

#### Configuration Method

Set your credentials in `.env`:

```bash
# Web service authentication configuration
WEB_USERNAME=admin
WEB_PASSWORD=admin123
```

#### Default Credentials

*   Username: `admin`
*   Password: `admin123`

**⚠️ Important: Change the default password in production!**

#### Authentication Scope

*   **Requires authentication:** All API endpoints, the Web UI, and static resources.
*   **No authentication:** Health check endpoint (`/health`)

#### Usage

1.  **Browser Access:** A login prompt will appear upon visiting the Web UI.
2.  **API Calls:** Authentication information must be included in request headers.
3.  **Frontend JavaScript:** Authentication handled automatically.

#### Security Recommendations

1.  Use strong passwords.
2.  Use HTTPS in production.
3.  Change credentials periodically.
4.  Restrict access by IP address through a firewall.

Detailed configuration is available in [AUTH_README.md](AUTH_README.md).

## FAQ

Find answers to common questions about setup, AI configuration, and anti-scraping strategies in the FAQ.

👉  **[Click Here to View the FAQ (FAQ.md)](FAQ.md)**

## Acknowledgements

Thanks to the projects referenced and contributions from the LinuxDo community, and ClaudeCode/Aider/Gemini for making it easier to write code.

## Support & Sponsoring

If this project helps you, consider showing support:

[Insert Alipay/WeChat Pay images as in the original README]

## Important Notes

*   Adhere to Goofish's user agreement and robots.txt rules to avoid server load or account restrictions.
*   This project is for educational and research purposes only; avoid illegal activities.
*   Released under the [MIT License](LICENSE).
*   The author is not responsible for any losses resulting from the use of this software.
*   See [DISCLAIMER.md](DISCLAIMER.md) for more information.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)