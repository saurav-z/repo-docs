# AI-Powered Xianyu (Goofish) Item Monitor

**Tired of missing out on the best deals on Xianyu?** This project is an intelligent Xianyu item monitor that leverages Playwright, AI analysis, and a user-friendly web interface to help you find the perfect items.  [View the original repository](https://github.com/dingyufei615/ai-goofish-monitor)

## Key Features:

*   ✅ **Intuitive Web UI:**  Manage tasks, edit AI criteria, view real-time logs, and filter results without touching the command line.
*   💬 **AI-Driven Task Creation:** Describe your desired item in natural language, and let AI create a monitoring task with complex filters.
*   ⚙️ **Multi-Tasking:** Monitor multiple keywords simultaneously via `config.json`, with independent execution.
*   ⚡ **Real-time Analysis:** Analyze new listings instantly, avoiding delays common with batch processing.
*   🧠 **Deep AI Analysis:** Integrate multimodal LLMs (e.g., GPT-4o) for comprehensive analysis based on item descriptions, images, and seller profiles.
*   🛠️ **Highly Customizable:** Configure keywords, price ranges, filtering conditions, and AI prompts for each task.
*   🔔 **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat Enterprise Robot, and [Bark](https://bark.day.app/) on your phone or desktop.
*   📅 **Scheduled Tasks:** Utilize cron expressions for flexible, automated task scheduling.
*   🐳 **Docker Deployment:**  Easy and standardized deployment with pre-configured `docker-compose` files.
*   🛡️ **Robust Anti-Detection:** Simulate human behavior with random delays and user actions to enhance stability.

## Screenshots:

**(Include the provided screenshots here, such as "后台任务管理," "后台监控截图," and "ntf通知截图."  Make sure to use markdown image syntax: `![alt text](image.png)`)**

## Getting Started:

### 1. Prerequisites:

*   **Python:** Python 3.10 or higher is recommended.
*   **Clone the repository:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```
*   **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuration:

1.  **Set Environment Variables:**  Create a `.env` file from `.env.example` and fill in your configuration:

    ```bash
    cp .env.example .env
    ```

    | Environment Variable | Description                                                     | Required | Notes                                                                                                                      |
    | :------------------- | :-------------------------------------------------------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------- |
    | `OPENAI_API_KEY`     | Your AI model service provider's API Key.                        | Yes      | May be optional for local or specific proxy services.                                                                        |
    | `OPENAI_BASE_URL`    | AI model API endpoint, compatible with OpenAI format.            | Yes      |  Specify the base path, e.g., `https://ark.cn-beijing.volces.com/api/v3/`.                                                       |
    | `OPENAI_MODEL_NAME`  | The specific model name you are using.                          | Yes      |  **Crucial:** Choose a multi-modal model that supports image analysis, such as `doubao-seed-1-6-250615` or `gemini-2.5-pro`.  |
    | `PROXY_URL`          | (Optional) HTTP/S proxy for bypassing geo-restrictions.         | No       | Supports `http://` and `socks5://`. Example: `http://127.0.0.1:7890`.                                                     |
    | ... (Other Variables) | (See the original README for more options like notification services and browser settings) |

    > 💡 **Debugging Tip:** If you encounter 404 errors with the AI API, try using the Ali or Volcano API for initial testing to ensure basic functionality.

2.  **Obtain Login Credentials (Essential!):**  You must provide valid login credentials for Xianyu.  The Web UI is the recommended way:

    **Web UI Method (Recommended):**
    1.  Start the Web UI (see step 3).
    2.  Go to "System Settings" in the UI.
    3.  Click "Manual Update" under "Login Status File."
    4.  Follow the instructions in the pop-up to install the Xianyu login state extension in Chrome, log in to the Xianyu website, extract the login state, and paste it into the Web UI.

    **Alternative Login Script (If you can run a browser on the server):**
    ```bash
    python login.py
    ```
    This will open a browser for you to log in via the Xianyu mobile app QR code.  It will create a `xianyu_state.json` file.

### 3.  Start the Web Server:

```bash
python web_server.py
```

### 4. Usage:

1.  Open the web UI at `http://127.0.0.1:8000`.
2.  Go to "Task Management" and click "Create New Task."
3.  Describe your item in natural language (e.g., "Looking for a used Sony A7M4 camera, mint condition, under $1300").
4.  The AI will generate a sophisticated analysis setup.
5.  Add scheduling or click "Start" to begin monitoring!

## Docker Deployment (Recommended):

### 1. Prerequisites (Similar to Local):

1.  **Install Docker Engine:** Ensure Docker is installed.
2.  **Clone and Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```
    Create your `.env` file (see above).
3.  **Get Login Credentials (Critical for Docker!):**

    *   **After** the container is running, access the Web UI at `http://127.0.0.1:8000`.
    *   Go to "System Settings" and click "Manual Update" under "Login Status File."
    *   Follow the prompts in the UI as described in the Web UI login method.

### 2. Run the Docker Container:

```bash
docker-compose up --build -d
```

### 3. Access and Manage:

*   **Web UI:**  `http://127.0.0.1:8000`
*   **Logs:** `docker-compose logs -f`
*   **Stop:** `docker-compose stop`
*   **Start:** `docker-compose start`
*   **Remove:** `docker-compose down`

## (Include the "Web UI Functionality at a Glance" section from the original README here.  Use appropriate Markdown formatting.)**

## (Include the "Working Process" diagram from the original README here.  Use Mermaid code.)**

## Frequently Asked Questions (FAQ):

**(Include the "Common Questions (FAQ)" section from the original README here.)**

## Acknowledgements:

**(Include the "致谢" section.)**

## Support & Sponsoring:

**(Include the "Support & Sponsoring" section.)**

## Important Notes:

*   Respect Xianyu's terms of service and `robots.txt`. Avoid excessive requests to prevent issues.
*   This project is for learning and research only. Do not use it for illegal purposes.
*   The project is licensed under the [MIT License](LICENSE).
*   Refer to the [免责声明](DISCLAIMER.md) for further information.