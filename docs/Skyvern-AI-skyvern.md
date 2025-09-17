<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png"/>
  </picture>
  <br />
  Skyvern: Automate Browser Workflows with AI 🐉
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

**Skyvern empowers you to automate complex browser-based tasks using the power of LLMs and computer vision, eliminating the need for brittle, website-specific scripts.**  See the original repo here: [https://github.com/Skyvern-AI/skyvern](https://github.com/Skyvern-AI/skyvern).

## Key Features

*   **AI-Powered Automation:** Leverage LLMs to understand and interact with websites, navigating and completing tasks without the need for custom code for each site.
*   **Resilient to Website Changes:** Built to adapt to website layout changes, ensuring your automation continues to function even as sites evolve.
*   **Cross-Website Compatibility:** Apply a single workflow across numerous websites, streamlining your automation efforts.
*   **Advanced Reasoning:** Benefit from LLMs that understand nuanced interactions, such as inferring answers to questions or recognizing similar products.
*   **Comprehensive Workflow Support:** Build complex automated processes with features including navigation, data extraction, loops, file handling, and more.
*   **Real-Time Monitoring:** Livestream browser viewport during tasks to observe Skyvern's actions.
*   **Form Filling & Data Extraction:** Seamlessly fill forms and extract data from websites with structured output.
*   **Authentication Support:** Automate tasks behind logins using a variety of authentication methods, including 2FA.
*   **Integrations:** Connect to other apps via Zapier, Make.com, and N8N.
*   **Model Context Protocol (MCP) Support**: Use any LLM that supports MCP.
*   **Broad LLM Support:** Compatible with OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Gemini, Ollama, OpenRouter, and OpenAI-compatible models.

## Quickstart

### Skyvern Cloud

Get started instantly with [Skyvern Cloud](https://app.skyvern.com), a managed service.

### Install & Run Locally

**Prerequisites:**

*   [Python 3.11.x](https://www.python.org/downloads/) (or 3.12)
*   [NodeJS & NPM](https://nodejs.org/en/download/)
*   (Windows only) [Rust](https://rustup.rs/) and VS Code with C++ dev tools and Windows SDK

**Installation:**

```bash
pip install skyvern
```

**Quick Start:**

```bash
skyvern quickstart
```

**Run a Task (UI):**

```bash
skyvern run all
```
Access the UI at http://localhost:8080.

**Run a Task (Code):**

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

See the full documentation at [https://docs.skyvern.com/](https://docs.skyvern.com/).

## How Skyvern Works

Skyvern leverages a swarm of agents to understand and interact with websites using a design inspired by BabyAGI and AutoGPT. The system diagrams shows the process of Skyvern working with a website. This approach provides:

*   **Adaptability:** Operates on websites it hasn't seen before, mapping visual elements to required actions.
*   **Resilience:** Resistant to website layout changes.
*   **Scalability:** Enables workflows to work on numerous websites.
*   **Intelligent Interactions:** Leverages LLMs to manage sophisticated interactions, such as dealing with inferred knowledge.

## Demo

![Invoice Downloading Demo](fern/images/invoice_downloading.gif)

## Performance & Evaluation

Skyvern demonstrates leading performance, including:

*   **SOTA Performance** on the [WebBench benchmark](webbench.ai) with 64.4% accuracy.
*   **Top Performer** on WRITE tasks (RPA-related tasks).

See evaluation results and a technical report here: [https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

## Advanced Usage

### Control Your Own Browser (Chrome)

```python
from skyvern import Skyvern

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

### Helpful Commands

```bash
# Launch the Skyvern Server Separately
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

1.  Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Clone the repository.
3.  Run `skyvern init llm`.
4.  Fill in the LLM provider key in [docker-compose.yml](./docker-compose.yml).
5.  Run `docker compose up -d`.
6.  Access the UI at `http://localhost:8080`.

## Skyvern Features

### Skyvern Tasks

Tasks are the fundamental units of work. They take a `url`, `prompt`, and optional parameters for data schema and error codes.

### Skyvern Workflows

Chain tasks together to create automated, comprehensive workflows.  Supported features include:

*   Navigation
*   Action
*   Data Extraction
*   Loops
*   File Parsing
*   File Uploads
*   Emailing
*   Text Prompts
*   Tasks
*   (Coming soon) Conditionals
*   (Coming soon) Custom Code Block

### Livestreaming

View real-time browser viewport interaction.

### Form Filling

Native form-filling capabilities.

### Data Extraction

Extract structured data based on your schema.

### File Downloading

Download and automatically store files.

### Authentication

Supports various authentication methods.

### 🔐 2FA Support (TOTP)

Supports various 2FA methods.

### Password Manager Integrations

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

## Real-world Examples

*   Invoice Downloading ([Book a demo](https://meetings.hubspot.com/skyvern/demo))
*   Job Application Automation ([💡 See it in action](https://app.skyvern.com/tasks/create/job_application))
*   Materials Procurement Automation ([💡 See it in action](https://app.skyvern.com/tasks/create/finditparts))
*   Government Website Navigation ([💡 See it in action](https://app.skyvern.com/tasks/create/california_edd))
*   Contact Form Filling ([💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms))
*   Insurance Quote Retrieval ([💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros))

    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)

## Documentation

Find detailed information on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

[Table of Supported LLMs and Environment Variables from the original README]

## Feature Roadmap

*   Open Source
*   Workflow support
*   Improved context
*   Cost Savings
*   Self-serve UI
*   Workflow UI Builder
*   Chrome Viewport streaming
*   Past Runs UI
*   Auto workflow builder ("Observer") mode
*   Prompt Caching
*   Web Evaluation Dataset
*   Improved Debug mode
*   Chrome Extension
*   Skyvern Action Recorder
*   Interactable Livestream
*   Integrate LLM Observability tools
*   Langchain Integration

## Contributing

We welcome contributions!  See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Opt-out of telemetry by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).  Anti-bot measures in our cloud offering.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)