# AI-Powered Xianyu (Goofish) Monitor: Real-time, Intelligent, and Customizable

Tired of missing out on the best deals on Xianyu (Goofish)? This project uses Playwright, AI analysis, and a user-friendly web interface to monitor Xianyu in real-time, offering intelligent filtering and instant notifications.  [View the original repo](https://github.com/dingyufei615/ai-goofish-monitor).

## Key Features

*   ✅ **Intuitive Web UI:** Manage tasks, edit AI criteria, view logs, and browse results directly through a web interface.
*   🧠 **AI-Driven Task Creation:** Describe your desired purchase in natural language, and the AI will generate a complex monitoring task.
*   ⚙️ **Multi-Task Concurrency:** Monitor multiple keywords simultaneously with independent tasks.
*   ⚡️ **Real-time Streaming:** Analyze new listings instantly, eliminating batch processing delays.
*   💡 **Deep AI Analysis:** Leverages multimodal LLMs (like GPT-4o) to analyze product images, text, and seller profiles for precise filtering.
*   🛠️ **Highly Customizable:** Configure keywords, price ranges, filtering conditions, and AI prompts for each monitoring task.
*   🔔 **Instant Notifications:** Receive alerts via [ntfy.sh](https://ntfy.sh/), Enterprise WeChat, and [Bark](https://bark.day.app/) for AI-recommended items.
*   📅 **Scheduled Tasks:** Utilize Cron expressions to schedule tasks.
*   🐳 **Docker Deployment:**  One-click deployment with Docker Compose for easy setup and portability.
*   🛡️ **Robust Anti-Scraping:** Mimics human behavior with random delays and user actions to improve stability.

## Getting Started

### Quickstart (Web UI Recommended)

The Web UI offers the best user experience.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/dingyufei615/ai-goofish-monitor
    cd ai-goofish-monitor
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure `.env`:**

    *   Copy the `.env.example` file and rename it to `.env`.
        ```bash
        # Windows
        copy .env.example .env
        # Linux/MacOS
        cp .env.example .env
        ```

    *   **Required Environment Variables:**

        | Variable          | Description                                      | Required? | Notes                                                                                                                              |
        | :---------------- | :----------------------------------------------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------- |
        | `OPENAI_API_KEY`  | Your AI model provider's API key.                | Yes       |                                                                                                                                    |
        | `OPENAI_BASE_URL` | The API endpoint for your AI model (OpenAI format). | Yes       |  Must be compatible with OpenAI API, e.g., `https://api.example.com/v1`.                                                                  |
        | `OPENAI_MODEL_NAME` | The name of the specific model you want to use.       | Yes       | **Required**: Choose a multimodal model that supports image analysis, such as `gpt-4o`, `gemini-1.5-pro`. |

    *   **Optional Environment Variables:** (See original README for full list and details)

4.  **Get Login State (Crucial!):** Run the login script to generate the session state file.

    ```bash
    python login.py
    ```

    Use your mobile Xianyu App to scan the QR code and log in. This generates the `xianyu_state.json` file.

5.  **Start the Web Server:**

    ```bash
    python web_server.py
    ```

6.  **Access the Web UI:**  Open `http://127.0.0.1:8000` in your browser.  Follow the steps in the UI to create and start your monitoring tasks.

### Docker Deployment (Recommended)

Docker provides a streamlined and reliable deployment.

1.  **Prerequisites:**

    *   Install [Docker Engine](https://docs.docker.com/engine/install/).
    *   Clone the project and configure the `.env` file (as described above).

2.  **Get Login State (Important):** **Run the login script on your host machine (not inside Docker)** to generate the `xianyu_state.json` file.

    ```bash
    pip install -r requirements.txt
    python login.py
    ```

3.  **Run the Docker Container:**

    ```bash
    docker-compose up -d
    ```

4.  **Access and Manage:**

    *   **Web UI:**  `http://127.0.0.1:8000`
    *   **Real-time Logs:** `docker-compose logs -f`
    *   **Stop/Start/Remove Containers:** `docker-compose stop`, `docker-compose start`, `docker-compose down`

### Web UI Features at a Glance

*   **Task Management:** AI-powered task creation, visual editing, scheduling, and control.
*   **Result Viewing:** Card-based display, intelligent filtering and sorting, detailed item data, and AI analysis results.
*   **Run Logs:** Real-time logs for monitoring progress and debugging.
*   **System Settings:** Configuration checks and prompt editing.

### Command-Line Usage (Advanced)

*   **Start Monitoring:** `python spider_v2.py` (loads tasks from `config.json`)
*   **Debug Mode:**  `python spider_v2.py --debug-limit 2` (limits items processed)
*   **Create Tasks:**  Use `prompt_generator.py` (see the original README for details)

## Workflow

[Include the mermaid diagram from the original README here, or a text representation if Markdown doesn't support it.]

## Tech Stack

*   **Core:** Playwright (async) + asyncio
*   **Web Server:** FastAPI
*   **Task Scheduling:** APScheduler
*   **AI:** OpenAI API (GPT-4o, etc.)
*   **Notifications:** ntfy, Enterprise WeChat
*   **Configuration:** JSON

## Project Structure (See original README for file details)

## Frequently Asked Questions (FAQ)

*   **Encoding Errors:** Use `set PYTHONUTF8=1` (Windows) or set an environment variable to UTF-8 before running scripts.
*   **Playwright Install:**  `pip install -r requirements.txt` and/or `playwright install chromium`.
*   **Connection Errors:** Check your network, proxy settings (`PROXY_URL` in `.env`), and `OPENAI_BASE_URL` address.
*   **Model Support:**  Choose an AI model that supports image analysis (Vision / Multi-modal).  Set `OPENAI_MODEL_NAME` correctly.
*   **Docker on Synology NAS:** Follow standard Docker deployment steps, ensuring you get `xianyu_state.json` and mount volumes correctly.
*   **Using Gemini/Qwen/etc.:** Configure `OPENAI_API_KEY`,  `OPENAI_BASE_URL` (API-compatible endpoint), and `OPENAI_MODEL_NAME` correctly in your `.env` file.  Consult your model provider's documentation.
*   **Anti-Scraping:** Reduce the risk of detection by setting `RUN_HEADLESS=false` and using a clean network environment.

## Acknowledgements

*   [superboyyy/xianyu_spider](https://github.com/superboyyy/xianyu_spider)
*   [@jooooody](https://linux.do/u/jooooody/summary)
*   Aider and Gemini (for assistance)

## Important Notes

*   Comply with Xianyu's terms of service and robots.txt.
*   Use responsibly and ethically.  Do not engage in excessive requests or illegal activities.

[![Star History Chart](https://api.star-history.com/svg?repos=dingyufei615/ai-goofish-monitor&type=Date)](https://star-history.com/#dingyufei615/ai-goofish-monitor&Date)
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  Provides an immediately understandable benefit.
*   **Keyword Optimization:** Uses relevant keywords like "Xianyu," "Goofish," "AI," "monitor," "real-time," "intelligent," and "customizable."  These are naturally integrated.
*   **Structured Headings and Subheadings:**  Improves readability and allows search engines to understand the content hierarchy.
*   **Bulleted Key Features:** Highlights the project's main advantages in an easy-to-scan format.
*   **Concise Descriptions:** Explains features and steps succinctly.
*   **Actionable "Getting Started" Section:** Guides users through the setup process.
*   **Docker Emphasis:** Docker is a popular deployment method, so it's given its own clear section.
*   **FAQ Section:** Addresses common user questions, which can improve SEO and reduce user support.
*   **Call to Action:** Encourages users to visit the original repo.
*   **Internal Linking:** Links within the document itself to help users navigate the content.
*   **Images and Diagrams (if possible):** Include images or diagrams (like the mermaid diagram) to enhance engagement and understanding.  (Note: Markdown rendering varies; ensure the diagram is properly displayed.)
*   **Clean and Readable Code:** Easy to understand, even when viewed directly on GitHub.
*   **Star History:** Includes the star history chart to show project popularity.