<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  Skyvern: Automate Browser Workflows with AI & Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

**Skyvern empowers you to automate complex browser-based tasks using the power of Large Language Models (LLMs) and computer vision, simplifying web automation and RPA.**  This innovative approach moves beyond brittle, code-dependent solutions by leveraging AI to understand and interact with websites.

[🚀 Explore the original repository here](https://github.com/Skyvern-AI/skyvern)

## Key Features

*   **AI-Powered Automation:** Utilize LLMs and computer vision to navigate and interact with websites.
*   **Robustness:** Designed to withstand website layout changes, unlike traditional automation methods.
*   **Cross-Website Compatibility:** Apply the same workflow across numerous websites with minimal code adjustments.
*   **Advanced Reasoning:** Leverage LLMs for intelligent decision-making, such as handling variations in form fields and product details.
*   **Real-time Monitoring:** Use livestreaming to see exactly what Skyvern is doing on the web for debugging and understanding.
*   **Form Filling:** Automate the completion of web forms.
*   **Data Extraction:** Easily extract specific data from websites.
*   **File Handling:** Download files and automatically upload them to block storage.
*   **Authentication Support:** Integrations with various authentication methods and password managers to automate tasks behind logins, with support for 2FA.
*   **Integration Friendly:** Seamlessly connects with Zapier, Make.com, and N8N for extended use cases.
*   **Model Context Protocol (MCP) Support**: Full MCP model support, including multimodal llms like gpt4-turbo, gpt-4o, and gpt4o-mini, from OpenAI, and Claude, and Gemini.

## Quickstart

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern (UI)

```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task

### 3. Run task (Code)

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

*   You can also run a task on different targets:

```python
from skyvern import Skyvern

# Run on Skyvern Cloud
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Local Skyvern service
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

### Skyvern Cloud
[Skyvern Cloud](https://app.skyvern.com) is a managed cloud version of Skyvern that allows you to run Skyvern without worrying about the infrastructure. It allows you to run multiple Skyvern instances in parallel and comes bundled with anti-bot detection mechanisms, proxy network, and CAPTCHA solvers.

## How Skyvern Works

Skyvern leverages a task-driven agent design, inspired by BabyAGI and AutoGPT, combined with browser automation. It uses a swarm of agents to:

1.  **Comprehend** a website.
2.  **Plan** actions.
3.  **Execute** those actions using libraries like Playwright.

**Advantages:**

*   Operates on unseen websites.
*   Resists website layout changes.
*   Applies workflows to multiple sites.
*   Uses LLMs for intelligent interactions.

## Demo

https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

## Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Performance"/>
</p>

## Advanced Usage

### Control your own browser (Chrome)

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

### Run Skyvern with any remote browser

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get consistent output schema from your run

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

### Helpful commands to debug issues

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

1.  Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine
2.  Make sure you don't have postgres running locally (Run `docker ps` to check)
3.  Clone the repository and navigate to the root directory
4.  Run `skyvern init llm` to generate a `.env` file. This will be copied into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml). *If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).*
6.  Run the following command via the commandline:

```bash
docker compose up -d
```

7.  Navigate to `http://localhost:8080` in your browser to start using the UI

>   **Important:** Only one Postgres container can run on port 5432 at a time. If you switch from the CLI-managed Postgres to Docker Compose, you must first remove the original container:

```bash
docker rm -f postgresql-container
```

If you encounter any database related errors while using Docker to run Skyvern, check which Postgres container is running with `docker ps`.

## Skyvern Features

### Skyvern Tasks

Tasks are the fundamental building block inside Skyvern. Each task is a single request to Skyvern, instructing it to navigate through a website and accomplish a specific goal.

Tasks require you to specify a `url`, `prompt`, and can optionally include a `data schema` (if you want the output to conform to a specific schema) and `error codes` (if you want Skyvern to stop running in specific situations).

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Screenshot"/>
</p>

### Skyvern Workflows

Workflows are a way to chain multiple tasks together to form a cohesive unit of work.

Supported workflow features include:
1. Navigation
1. Action
1. Data Extraction
1. Loops
1. File parsing
1. Uploading files to block storage
1. Sending emails
1. Text Prompts
1. Tasks (general)
1. (Coming soon) Conditionals
1. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Workflow Example"/>
</p>

### 🔐 2FA Support (TOTP)
Skyvern supports a number of different 2FA methods to allow you to automate workflows that require 2FA.

Examples include:
1. QR-based 2FA (e.g. Google Authenticator, Authy)
1. Email based 2FA
1. SMS based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations
Skyvern currently supports the following password manager integrations:
- [x] Bitwarden
- [ ] 1Password
- [ ] LastPass

## Real-world Examples

*   Invoice Downloading
*   Job Application Automation
*   Materials Procurement
*   Government Website Automation
*   Contact Form Filling
*   Insurance Quote Retrieval

## Contributor Setup

```bash
pip install -e .
```

Set up your development environment to use pre-commit:

```bash
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` in your browser to start using the UI.
    *   The Skyvern CLI supports Windows, WSL, macOS, and Linux environments.

## Documentation

Detailed documentation is available on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

See the original README for a table of supported LLMs and their configurations.

## Feature Roadmap

*   Open Source
*   Workflow Support
*   Improved Context
*   Cost Savings
*   Self-Serve UI
*   Workflow UI Builder
*   Chrome Viewport Streaming
*   Past Runs UI
*   Auto Workflow Builder ("Observer") Mode
*   Prompt Caching
*   Web Evaluation Dataset
*   Improved Debug Mode
*   Chrome Extension
*   Skyvern Action Recorder
*   Interactable Livestream
*   Integrate LLM Observability tools
*   Langchain Integration

## Contributing

We welcome contributions! Please see the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Telemetry

By Default, Skyvern collects basic usage statistics to help us understand how Skyvern is being used. If you would like to opt-out of telemetry, please set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).