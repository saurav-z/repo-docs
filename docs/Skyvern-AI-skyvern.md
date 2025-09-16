<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  Skyvern: Automate Your Browser Workflows with AI
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Follow on Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="Follow on LinkedIn"/></a>
</p>

**Skyvern empowers you to automate browser-based tasks using the power of Large Language Models (LLMs) and computer vision, making complex web interactions simple.**

**Key Features:**

*   🌐 **Website Automation:** Automate tasks on any website without custom scripting, adapting to layout changes.
*   🧠 **AI-Powered Navigation:** Utilizes LLMs to understand and interact with web elements, mimicking human behavior.
*   ⚙️ **Workflow Automation:** Chain multiple tasks to automate more complex operations such as invoice downloading, and job application.
*   💻 **Form Filling & Data Extraction:**  Easily fill forms and extract structured data from websites.
*   📺 **Livestreaming & Debugging:**  View live browser sessions for debugging and understanding automation.
*   ✅ **Authentication Support:**  Integrates with various authentication methods, including 2FA.
*   🚀 **Integrations:**  Supports Zapier, Make.com, and N8N for seamless connection to other apps.

Ready to get started?  [Explore the Skyvern Repository on GitHub](https://github.com/Skyvern-AI/skyvern)

---

## Getting Started

### Quickstart

1.  **Install Skyvern:**

    ```bash
    pip install skyvern
    ```

2.  **Run Skyvern (for initial setup):**

    ```bash
    skyvern quickstart
    ```

3.  **Run a Task:**

    *   **UI (Recommended):**

        ```bash
        skyvern run all
        ```

        Then, go to `http://localhost:8080` to run a task using the UI.
    *   **Code:**

        ```python
        from skyvern import Skyvern

        skyvern = Skyvern()
        task = await skyvern.run_task(prompt="Find the top post on hackernews today")
        print(task)
        ```

        View task history at `http://localhost:8080/history`.  You can also specify `base_url` or `api_key` to use a local or cloud service.

### Dependencies

*   [Python 3.11.x](https://www.python.org/downloads/) (works with 3.12, not yet ready for 3.13)
*   [NodeJS & NPM](https://nodejs.org/en/download/)
*   For Windows: [Rust](https://rustup.rs/), VS Code with C++ dev tools and Windows SDK

---

## How Skyvern Works

Skyvern is built on the principles of task-driven autonomous agents, inspired by projects like BabyAGI and AutoGPT. It leverages browser automation with libraries like Playwright. Key advantages include:

*   Operates on unseen websites without custom code.
*   Resists website layout changes.
*   Applies workflows across various websites.
*   Uses LLMs for intelligent interaction and complex scenario handling.

For details, see our technical report: [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

---

## Demo

<!--
Demo content could be added, either inline or as a link to an external demo, like a video.
-->

## Performance & Evaluation

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai), with 64.4% accuracy.

*   **WRITE Task Performance:** Skyvern excels in WRITE tasks, such as form filling, login, and file downloads.

---

## Advanced Usage

*   **Control Your Browser:**  Specify the path to your Chrome executable (see examples in the original README).  Requires setting `CHROME_EXECUTABLE_PATH` and `BROWSER_TYPE=cdp-connect` in your `.env` file.
*   **Run with a Remote Browser:** Use a CDP connection URL.
*   **Consistent Output Schema:**  Define a `data_extraction_schema` in your prompt to structure the output.
*   **Debugging commands**:  `skyvern run server`, `skyvern run ui`, `skyvern status`, `skyvern stop all`, `skyvern stop ui`, `skyvern stop server`

---

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Clone the repository.
3.  Run `skyvern init llm` to create a `.env` file.
4.  Fill in your LLM provider key in `docker-compose.yml`.  (Important:  Configure correct server IP for the UI container if running on a remote server.)
5.  Run:

    ```bash
    docker compose up -d
    ```

6.  Access the UI at `http://localhost:8080`.
7.  **Important:** Remove the original Postgres container if you switch from CLI-managed Postgres to Docker Compose.

---

## Skyvern Features

*   **Skyvern Tasks:** Core building blocks – each task instructs Skyvern to navigate and complete a specific goal. Requires `url`, `prompt`, and optional `data schema` and `error codes`.
*   **Skyvern Workflows:** Chain multiple tasks. Supported features include navigation, actions, data extraction, loops, file handling, email sending, text prompts, and tasks.  (Coming soon: conditionals, custom code blocks).
*   **Livestreaming:**  View browser activity in real-time.
*   **Form Filling:** Native form filling capabilities.
*   **Data Extraction:** Extract data based on prompt or specified schema.
*   **File Downloading:** Automated file downloads with block storage upload.
*   **Authentication:** Supports various authentication methods, including 2FA and password manager integrations.

---

## Real-World Examples

*   Invoice Downloading (demo)
*   Job Application Automation
*   Automating materials procurement for manufacturing
*   Navigating government websites
*   Filling Contact Us Forms
*   Retrieving Insurance Quotes

---

## Documentation

Find detailed documentation on our [docs page](https://docs.skyvern.com). Contact us via email at [founders@skyvern.com] or Discord at [discord](https://discord.gg/fG2XXEuQX3) if you have any questions.

---

## Supported LLMs

(Table of supported LLMs and variables, as in the original README.)

---

## Feature Roadmap

(Copy from the original README)

---

## Contributing

We welcome contributions! Refer to our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

---

## Telemetry

Skyvern collects basic usage statistics. Opt-out by setting `SKYVERN_TELEMETRY=false`.

---

## License

Skyvern's core codebase is under the [AGPL-3.0 License](LICENSE), with the exception of anti-bot measures in our managed cloud. Contact us at [support@skyvern.com] for licensing inquiries.

---

## Star History

(The Star History chart as in the original README.)