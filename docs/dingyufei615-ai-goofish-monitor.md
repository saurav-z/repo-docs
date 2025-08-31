# AI-Powered Goofish Monitor: Smartly Track & Analyze Xianyu (Goofish) Listings 🎣

**Effortlessly monitor Xianyu (Goofish) for your desired items using AI, with a user-friendly web interface for streamlined management and instant notifications.** ([Original Repo](https://github.com/dingyufei615/ai-goofish-monitor))

## ✨ Key Features

*   **Intuitive Web UI:** Manage tasks, edit AI criteria, view logs, and filter results visually.
*   **AI-Driven Task Creation:** Describe your needs in natural language to create smart monitoring tasks.
*   **Concurrent Multi-Tasking:** Monitor multiple keywords simultaneously without interference via `config.json`.
*   **Real-time Processing:** Analyze new listings instantly, avoiding delays.
*   **Deep AI Analysis:** Leverage multimodal LLMs (like GPT-4o) to analyze listings, images, and seller profiles.
*   **Highly Customizable:** Configure keywords, price ranges, filters, and AI prompts for each task.
*   **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat group bots, and [Bark](https://bark.day.app/).
*   **Scheduled Tasks:** Utilize Cron expressions for automated task execution.
*   **Docker Deployment:** Deploy quickly and reliably with provided `docker-compose` configuration.
*   **Robust Anti-Scraping:** Employs realistic browser behavior and delays to enhance stability.

## 🖼️ Screenshots

*   **Task Management:**
    ![img.png](static/img.png)

*   **Monitoring View:**
    ![img_1.png](static/img_1.png)

*   **Notification Example:**
    ![img_2.png](static/img_2.png)

## 🚀 Getting Started (Web UI Recommended)

The web UI provides the best user experience for this project.

### Step 1: Environment Setup

> ⚠️ **Python Version:** Python 3.10 or higher is recommended for local deployment. Lower versions may cause installation or runtime errors.

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

1.  **Configure Environment Variables:** Copy `.env.example` to `.env` and edit it.

    **Windows:**

    ```cmd
    copy .env.example .env
    ```

    **Linux/macOS:**

    ```bash
    cp .env.example .env
    ```

    Environment variables:

    | Variable          | Description                                           | Required? | Notes                                                                                                                                                            |
    | :---------------- | :---------------------------------------------------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`  | Your AI model provider's API key.                     | Yes       | May be optional for local or specific proxy services.                                                                                                      |
    | `OPENAI_BASE_URL` | AI model API endpoint (OpenAI compatible).            | Yes       | Fill in the base path, e.g., `https://ark.cn-beijing.volces.com/api/v3/`.                                                                                 |
    | `OPENAI_MODEL_NAME`| Specific model name.                                 | Yes       | **Must** select a multimodal model, such as `doubao-seed-1-6-250615` or `gemini-2.5-pro`.                                                                |
    | `PROXY_URL`       | (Optional) HTTP/S proxy for bypassing geo-restrictions. | No        | Supports `http://` and `socks5://`, e.g., `http://127.0.0.1:7890`.                                                                                     |
    | ... (rest of the variables are here, as in the original README)

    > 💡 **Debugging Tip:** If you encounter 404 errors when configuring your AI API, try using APIs from Alibaba Cloud or Volcano Engine for initial testing.

    > 🔐 **Security Note:** The web interface uses Basic authentication.  The default username/password are `admin` / `admin123`.  **Change these in production!**

2.  **Get Login State (Crucial!)**: The crawler needs valid login credentials. We recommend using the Web UI:

    **Recommended: Through Web UI**

    1.  Skip this step and start the Web service in Step 3.
    2.  Open the Web UI, go to the "System Settings" page.
    3.  Find the "Login State File" and click the "Manual Update" button.
    4.  Follow the instructions:
        - Install the [Xianyu Login State Extractor](https://chromewebstore.google.com/detail/xianyu-login-state-extrac/eidlpfjiodpigmfcahkmlenhppfklcoa) Chrome extension.
        - Open and log in to the Xianyu website.
        - Click the extension icon and then "Extract Login State."
        - Click "Copy to Clipboard."
        - Paste the content into the Web UI and save.

    **Alternative: Run Login Script** (For local/desktop-enabled servers)

    ```bash
    python login.py
    ```

    Follow the prompts to log in via the Xianyu mobile app. This will generate `xianyu_state.json`.

### Step 3: Start the Web Server

Run the Web UI server after setting up the environment.

```bash
python web_server.py
```

### Step 4: Using the Application

Open `http://127.0.0.1:8000` in your browser.

1.  In the "Task Management" page, click "Create New Task."
2.  Describe your desired items using natural language (e.g., "Looking for a Sony A7M4 camera, 95% new or better, budget under 13,000, shutter count below 5000").
3.  The AI will automatically generate the task parameters.  Enter task name, keywords, and then create the task.
4.  Start or schedule the task from the main UI.

## 🐳 Docker Deployment (Recommended)

Docker provides a simplified and consistent deployment experience.

### Step 1: Environment Setup

1.  **Install Docker**: Ensure [Docker Engine](https://docs.docker.com/engine/install/) is installed.
2.  **Clone and Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Create `.env`:** Create and populate the `.env` file as described in **[Getting Started](#-getting-started-web-ui-recommended)**.
4.  **Get Login State (Essential!)**:  Because you cannot use the QR code scanner on the server, you *must* set the login state *after* starting the container through the Web UI:
    1.  Start the container with `docker-compose up -d` (on the host).
    2.  Open the Web UI in your browser (`http://127.0.0.1:8000`).
    3.  Go to the "System Settings" page and click the "Manual Update" button.
    4.  Follow the instructions to obtain and paste the login information.

> ℹ️ **Python Version Note**: The Docker image uses Python 3.11.

### Step 2: Run Docker Container

1.  Run `docker-compose up --build -d` in the project directory.

If you encounter network issues in the container, configure the network and DNS settings as necessary.

### Step 3: Access and Manage

-   **Access Web UI:** Open `http://127.0.0.1:8000` in your browser.
-   **View Logs:** `docker-compose logs -f`
-   **Stop Container:** `docker-compose stop`
-   **Start Stopped Container:** `docker-compose start`
-   **Stop and Remove Container:** `docker-compose down`

## 📸 Web UI Feature Overview

*   **Task Management:**
    *   AI-powered task creation using natural language.
    *   Edit task parameters, start/stop, and delete tasks in the UI.
    *   Cron job scheduling.
*   **Result Viewing:**
    *   Card-based display of matching items.
    *   Smart filtering (recommended items) and sorting.
    *   Detailed views of item data and AI analysis results.
*   **Real-time Logs:**
    *   Live logs for troubleshooting.
    *   Log management (refresh, clear).
*   **System Settings:**
    *   Status checks for environment variables and login status.
    *   Prompt editing for real-time adjustment of AI logic.

## 🚀 Workflow Diagram

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

## 🔐 Web Interface Authentication

### Configuration

The web interface uses Basic authentication.

#### Configuration Method

Set credentials in the `.env` file:

```bash
# Web Service Authentication
WEB_USERNAME=admin
WEB_PASSWORD=admin123
```

#### Default Credentials

*   Username: `admin`
*   Password: `admin123`

**⚠️ Important: Change defaults in production!**

#### Authentication Scope

*   **Requires Auth:** All API endpoints, web interface, static resources
*   **No Auth Required:** Health check endpoint (`/health`)

#### Usage

1.  **Browser Access:** Authentication prompt when accessing the web interface.
2.  **API Calls:** Requires Basic authentication in request headers.
3.  **Frontend JavaScript:** Automatically handles authentication.

#### Security Recommendations

1.  Strong passwords
2.  HTTPS in production
3.  Regular credential changes
4.  Restrict IP access through firewall

## ❓ FAQ

Comprehensive FAQ: **[Click to view FAQ (FAQ.md)](FAQ.md)**

## Acknowledgements

Project inspiration and code contributions from:

*   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)

*   [@jooooody](https://linux.do/u/jooooody/summary) and the LinuxDo community.

*   ClaudeCode/ModelScope/Gemini for their models/tools.

## Disclaimer

*   Follow Xianyu's rules and `robots.txt`.
*   For learning and research only, not illegal use.
*   MIT License.
*   No liability for damages.
*   See [Disclaimer](DISCLAIMER.md) for details.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)