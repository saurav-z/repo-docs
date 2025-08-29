<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
    </picture>
  </a>
  <br />
  <br />
  Automate Your Browser Workflows with AI: Introducing Skyvern 🐉
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

Skyvern is a powerful, open-source tool that uses Large Language Models (LLMs) and computer vision to automate complex browser-based tasks, eliminating the need for brittle and website-specific scripts. [Explore Skyvern on GitHub](https://github.com/Skyvern-AI/skyvern).

**Key Features:**

*   **AI-Powered Automation:**  Intelligently navigates and interacts with websites using LLMs, adapting to website changes.
*   **Workflow Creation:** Build complex, multi-step automation workflows.
*   **Data Extraction & Form Filling:** Extract structured data and automate form submissions.
*   **Livestreaming & Debugging:**  Visually track and debug workflows in real-time.
*   **Advanced Integrations:** Supports 2FA (including TOTP, QR, Email & SMS), Password Managers (Bitwarden), and Integrations (Zapier, Make.com, N8N).
*   **Open Source:**  Benefit from the community, customize to your needs, and have full control over your automations.
*   **SOTA Performance:** Achieves state-of-the-art performance on WebBench benchmark and is the best performing agent on WRITE tasks.

### Quickstart

**Installation:**

```bash
pip install skyvern
```

**Run:**

```bash
skyvern quickstart
```
Then start UI
```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task.

**Code Example:**

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

### Skyvern Cloud

For a fully managed solution, try [Skyvern Cloud](https://app.skyvern.com). It offers parallel instance execution, anti-bot mechanisms, and CAPTCHA solvers.

<br>

## How Skyvern Works

Skyvern uses a swarm of agents and browser automation (Playwright) to comprehend a website, plan, and execute actions.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" />
</picture>

This approach offers:

1.  **Website Agnosticism:** Operates on unseen websites by mapping visual elements to actions.
2.  **Resilience to Layout Changes:**  Adaptable to website updates due to its reliance on LLMs for interaction, rather than hard-coded selectors.
3.  **Scalability:** Apply a single workflow across many websites, as the system can reason through interactions.
4.  **Advanced Reasoning:** Leverages LLMs to handle complex scenarios, such as inferring answers to form questions or understanding product equivalency across different websites.

For a detailed technical analysis, see our report: [Skyvern 2.0 Technical Report](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

<br>

## Performance & Evaluation

Skyvern excels in browser automation tasks and delivers impressive results:

*   **WebBench Benchmark:** Skyvern has SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.
*   **WRITE Task Performance:**  Leading performance in tasks that involve form filling, logins, and file downloads.

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench WRITE Task Performance"/>
</p>

<br>

## Demo

See Skyvern in action:

[https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

<br>

## Advanced Usage

### Control Your Own Browser (Chrome)

```python
from skyvern import Skyvern

# The path to your Chrome browser. This example path is for Mac.
browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
skyvern = Skyvern(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY",
    browser_path=browser_path,
)
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Run with any remote browser

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Consistent Output Schema

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
    data_extraction_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the top post"
            },
            "url": {
                "type": "string",
                "description": "The URL of the top post"
            },
            "points": {
                "type": "integer",
                "description": "Number of points the post has received"
            }
        }
    }
)
```

### Helpful Commands to Debug

```bash
# Launch the Skyvern Server Separately*
skyvern run server

# Launch the Skyvern UI
skyvern run ui

# Check status of the Skyvern service
skyvern status

# Stop the Skyvern service
skyvern stop all

# Stop the Skyvern UI
skyvern stop ui

# Stop the Skyvern Server Separately
skyvern stop server
```

## Docker Compose Setup

Follow these steps to set up Skyvern using Docker Compose.

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Ensure no local Postgres instance is running (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file.
5.  Fill in your LLM provider key in `docker-compose.yml`.  **Important:** set the UI container's server IP if running on a remote server.
6.  Run: `docker compose up -d`.
7.  Access the UI at `http://localhost:8080`.

**Important:** Remove the original container if you switch from the CLI-managed Postgres to Docker Compose.
```bash
docker rm -f postgresql-container
```

## Skyvern Features

*   **Skyvern Tasks:** Instructions to navigate and accomplish specific goals on a website.
*   **Skyvern Workflows:** Chain multiple tasks for cohesive automation units.
*   **Livestreaming:**  Stream browser viewport for debugging.
*   **Form Filling:**  Automated form input.
*   **Data Extraction:** Extract data using a specified schema.
*   **File Downloading:** Automatic file download and upload to block storage.
*   **Authentication:** Supports multiple authentication methods.
    *   **2FA:** Support for various 2FA methods.  Learn more [here](https://docs.skyvern.com/credentials/totp).
    *   **Password Manager Integrations:** Supports Bitwarden.

<br>

## Real-World Examples

*   **Invoice Downloading:** Automated downloading of invoices from various websites.
*   **Job Application Automation:**  Automating the job application process.
*   **Material Procurement:** Automating the procurement process for manufacturing.
*   **Government Website Automation:** Automating account creation and form-filling on government sites.
*   **Contact Form Filling:** Automating the process of filling out contact us forms.
*   **Insurance Quote Retrieval:** Retrieving insurance quotes from various providers.

<br>

## Contributor Setup

For a complete local environment CLI Installation.
```bash
pip install -e .
```
The following command sets up your development environment to use pre-commit (our commit hook handler)
```
skyvern quickstart contributors
```

1.  Access the UI at `http://localhost:8080`.

    *The Skyvern CLI supports Windows, WSL, macOS, and Linux.*

<br>

## Documentation

Find detailed documentation on our [📕 docs page](https://docs.skyvern.com).

<br>

## Supported LLMs

Skyvern supports a wide range of LLMs from various providers:

*   OpenAI
*   Anthropic
*   Azure OpenAI
*   AWS Bedrock
*   Gemini
*   Ollama
*   OpenRouter
*   OpenAI-compatible

### Environment Variables

Detailed environment variable configurations are provided in the original README.

<br>

## Feature Roadmap

Our upcoming features include:

*   Improved Debug Mode.
*   Chrome Extension.
*   Skyvern Action Recorder.
*   Interactable Livestream.
*   LLM Observability tools integration.
*   Langchain integration

<br>

## Contributing

We welcome your contributions!  Refer to our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).
If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

<br>

## Telemetry

By Default, Skyvern collects basic usage statistics to help us understand how Skyvern is being used. If you would like to opt-out of telemetry, please set the `SKYVERN_TELEMETRY` environment variable to `false`.

<br>

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).

<br>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)