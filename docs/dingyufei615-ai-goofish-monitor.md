# AI-Powered Goofish (Xianyu) Smart Monitoring Robot

**Automatically monitor and analyze Xianyu (Goofish) listings with AI, delivering real-time alerts and insights through a user-friendly web interface.** ([Original Repo](https://github.com/dingyufei615/ai-goofish-monitor))

## Key Features:

*   **Intuitive Web UI:** Manage tasks, edit AI criteria, view real-time logs, and filter results through a complete web interface.
*   **AI-Driven Task Creation:** Create new monitoring tasks with complex filtering logic using natural language descriptions of your desired item.
*   **Concurrent Monitoring:** Monitor multiple keywords simultaneously using the `config.json` file, with each task operating independently.
*   **Real-time Processing:** Analyze new listings immediately upon discovery, eliminating batch processing delays.
*   **Deep AI Analysis:** Utilize multimodal large language models (e.g., GPT-4o) to deeply analyze item descriptions, images, and seller profiles for accurate filtering.
*   **Highly Customizable:** Configure independent keywords, price ranges, filters, and AI analysis prompts for each monitoring task.
*   **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), WeChat group bots, and [Bark](https://bark.day.app/) to your phone or desktop.
*   **Scheduled Tasks:** Schedule tasks using Cron expressions for automated execution.
*   **Docker Deployment:**  Quick and standardized deployment with provided `docker-compose` configurations.
*   **Robust Anti-Scraping:**  Simulates human behavior with random delays and user actions to enhance stability.

## Screenshots:

**(Illustrative screenshots from the original README here)**

## Getting Started:

This section outlines how to set up the project, including Python, dependencies, and configuration steps.

### Step 1: Environment Setup

1.  **Python Version**:  Requires Python 3.10 or higher to avoid dependency installation failures and runtime errors (like `ModuleNotFoundError: No module named 'PIL'`).
2.  **Clone the repository**:

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Configuration

1.  **Configure Environment Variables**: Copy `.env.example` to `.env` and populate the necessary variables:

    *   `OPENAI_API_KEY`: Your AI model service API key.
    *   `OPENAI_BASE_URL`:  Base URL for your AI model's API (compatible with OpenAI format).
    *   `OPENAI_MODEL_NAME`: Name of the multimodal model you want to use (e.g., `doubao-seed-1-6-250615`, `gemini-2.5-pro`).
    *   `PROXY_URL` (Optional): Proxy for access to the internet.
    *   Notification services and Webhook configurations, as described in the original README.
    *   Other settings include login browser configurations, headless mode, and server details.

>   **Debugging Tip:**  If you encounter 404 errors when configuring the AI API, first test with a provider like AliCloud or VolcEngine to ensure basic functionality before trying other providers.

>   **Security Note:**  The web interface uses Basic Authentication. Change the default username/password (`admin`/`admin123`) in production environments!

2.  **Obtain Login Status (Crucial)**: Provide valid login credentials for the Xianyu platform.  The recommended method is using the Web UI:

    *   Start the web server (Step 3).
    *   Access "System Settings" in the Web UI.
    *   Click the "Manual Update" button under "Login Status File".
    *   Follow the instructions to install the Xianyu login state extension.
    *   Login to Xianyu on the browser, extract, and copy the login information from the browser extension, and paste it into the Web UI.

    The legacy alternative is to run `login.py` in a terminal.

### Step 3: Start the Web Server

```bash
python web_server.py
```

### Step 4: Usage

Access the management interface at `http://127.0.0.1:8000`:

1.  Go to **"Task Management"** and click **"Create New Task"**.
2.  Use natural language to describe your desired item (e.g., "Looking for a used Sony A7M4 camera, budget under $1300...").
3.  Click Create; the AI generates analysis criteria.
4.  Start or schedule your task.

## Docker Deployment

Deploy using Docker for a simplified setup.

### Step 1: Environment Preparation

1.  **Install Docker**.
2.  **Clone and Configure:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```
    Configure `.env` as described above.
3.  **Obtain Login Status**. After running Docker Compose, access the Web UI as described in Step 2 to configure login status.

### Step 2: Run Docker Container

Run from the root project directory:

```bash
docker-compose up --build -d
```

### Step 3: Access and Manage

*   Access the Web UI at `http://127.0.0.1:8000`.
*   View logs: `docker-compose logs -f`
*   Stop: `docker-compose stop`
*   Start: `docker-compose start`
*   Remove: `docker-compose down`

## Web UI Features:

**(Full Feature List as in the original README)**

## Workflow:

**(Workflow Diagram as in the original README)**

## Authentication Details:

**(Authentication details as in the original README)**

## FAQ:

**(Link to FAQ.md as in the original README)**

## Acknowledgements

**(Acknowledgements as in the original README)**

## Considerations:

**(Considerations as in the original README)**

## License

This project is released under the [MIT License](LICENSE).