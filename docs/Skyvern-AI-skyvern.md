<h1 align="center">
  Skyvern: Automate Browser Workflows with LLMs and Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

> Unleash the power of AI to automate complex browser-based tasks with Skyvern, a cutting-edge solution leveraging LLMs and computer vision.

Skyvern ([Original Repo](https://github.com/Skyvern-AI/skyvern)) revolutionizes browser automation, offering a powerful API for automating manual workflows across a wide array of websites, surpassing limitations of traditional methods. Skyvern leverages advanced LLMs and computer vision to interact with websites, enabling intelligent, resilient, and adaptable automation.

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern in action"/>
</p>

## Key Features

*   **Intelligent Automation:** Uses Vision LLMs to understand and interact with websites, adapting to changes without requiring code modifications.
*   **Workflow Versatility:** Handles tasks on websites it has never encountered before.
*   **Resilient to Layout Changes:** No reliance on hardcoded selectors (e.g. XPath) makes it less brittle.
*   **Cross-Website Compatibility:**  Applies a single workflow across numerous sites.
*   **Advanced Reasoning:** Leverages LLMs to handle complex scenarios (e.g. fill out insurance quotes).
*   **Skyvern Cloud:** Try out a managed cloud version that includes features like anti-bot detection and CAPTCHA solvers, available at [app.skyvern.com](https://app.skyvern.com)
*   **Livestreaming:** Real-time browser viewport streaming for debugging.
*   **Form Filling:** Native support for completing forms.
*   **Data Extraction:** Extracting specific data from websites, with structured outputs.
*   **File Downloading:** Seamless file downloads with automatic block storage integration.
*   **Authentication Support:** Integrations for various authentication methods, including 2FA and password managers.
*   **Model Context Protocol (MCP):** Supports Model Context Protocol (MCP) to allow you to use any LLM that supports MCP.
*   **Integrations:** Supports Zapier, Make.com, and N8N for integration with other apps.

## Quickstart

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern

```bash
skyvern quickstart
```

### 3. Run Task

#### UI (Recommended)

```bash
skyvern run all
```

Access the UI at http://localhost:8080 and execute your tasks through the interface.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

*   Skyvern launches a browser to execute the task, closing it upon completion.
*   View task history at http://localhost:8080/history.

You can also specify your execution target:
```python
from skyvern import Skyvern

# Run on Skyvern Cloud
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Local Skyvern service
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## How it Works

Skyvern is inspired by the Task-Driven autonomous agent design popularized by [BabyAGI](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT). It gives Skyvern the ability to interact with websites using browser automation libraries like [Playwright](https://playwright.dev/).

Skyvern uses a swarm of agents to comprehend a website, and plan and execute its actions:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" />
</picture>

For a detailed technical overview, check out the report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<!-- Redo demo -->
[Check out the demo](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

## Performance & Evaluation

Skyvern achieves industry-leading performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy rate.

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

## Performance on WRITE Tasks

Skyvern is the best-performing agent for WRITE tasks (RPA adjacent tasks).

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench WRITE Task Performance"/>
</p>

## Advanced Usage

### Control Your Browser
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

### Run with a Remote Browser
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

### Helpful Debug Commands

```bash
skyvern run server        # Launch Skyvern Server Separately
skyvern run ui            # Launch Skyvern UI
skyvern status            # Check Skyvern service status
skyvern stop all          # Stop all Skyvern services
skyvern stop ui           # Stop Skyvern UI
skyvern stop server       # Stop Skyvern Server Separately
```

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Make sure you don't have postgres running locally (Run `docker ps` to check).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file and copy it into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml).  If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).
6.  Run: `docker compose up -d`
7.  Access the UI at `http://localhost:8080`.

> **Important:** If using Docker Compose, remove the original CLI-managed Postgres container if necessary: `docker rm -f postgresql-container`

## Skyvern Features - Deep Dive

### Tasks
Tasks are the fundamental building block inside Skyvern. Each task is a single request to Skyvern, instructing it to navigate through a website and accomplish a specific goal.

Tasks require you to specify a `url`, `prompt`, and can optionally include a `data schema` (if you want the output to conform to a specific schema) and `error codes` (if you want Skyvern to stop running in specific situations).

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Example"/>
</p>

### Workflows
Workflows combine multiple tasks for complex operations.

Examples:
*   Download invoices newer than a specific date.
*   Automate product purchases from an e-commerce store.

Supported features include:
1.  Navigation
2.  Action
3.  Data Extraction
4.  Loops
5.  File parsing
6.  Uploading files to block storage
7.  Sending emails
8.  Text Prompts
9.  Tasks (general)
10. (Coming soon) Conditionals
11. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Invoice Downloading Workflow Example"/>
</p>

### Authentication

Skyvern supports multiple authentication methods to streamline automated tasks:

<p align="center">
  <img src="fern/images/secure_password_task_example.png" alt="Secure Password Task Example"/>
</p>

*   🔐 2FA (TOTP) Support
*   Password Manager Integrations:
    *   ✅ Bitwarden
    *   ⬜ 1Password
    *   ⬜ LastPass

## Real-world Examples

Explore how Skyvern automates real-world workflows.

### Invoice Downloading
[Book a demo](https://meetings.hubspot.com/skyvern/demo)
<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo"/>
</p>

### Job Application Process
[💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Demo"/>
</p>

### Materials Procurement
[💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Finditparts Recording Crop"/>
</p>

### Government Website Navigation
[💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
<p align="center">
  <img src="fern/images/edd_services.gif" alt="EDD Services Demo"/>
</p>

### Contact Us Forms
[💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Forms Demo"/>
</p>

### Insurance Quote Retrieval
[💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="BCI Seguros Demo"/>
</p>

[💡 See it in action](https://app.skyvern.com/tasks/create/geico)
<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Demo"/>
</p>

## Contributor Setup

For a complete local environment CLI Installation
```bash
pip install -e .
```
Set up your development environment to use pre-commit:
```
skyvern quickstart contributors
```

Access the UI by navigating to http://localhost:8080 in your browser.

*The Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Find detailed documentation on our [📕 docs page](https://docs.skyvern.com).  Contact us with any questions [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider | Supported Models |
| -------- | ------- |
| OpenAI   | gpt4-turbo, gpt-4o, gpt-4o-mini |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Azure OpenAI | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Gemini | Gemini 2.5 Pro and flash, Gemini 2.0 |
| Ollama | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter | Access models through [OpenRouter](https://openrouter.ai) |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

## Environment Variables

### OpenAI

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OPENAI` | Enable OpenAI models | Boolean | `true`, `false` |
| `OPENAI_API_KEY` | OpenAI API Key | String | `sk-1234567890` |
| `OPENAI_API_BASE` | OpenAI API Base (optional) | String | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID (optional) | String | `your-org-id` |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

### Anthropic

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_ANTHROPIC` | Enable Anthropic models | Boolean | `true`, `false` |
| `ANTHROPIC_API_KEY` | Anthropic API key | String | `sk-1234567890` |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

### Azure OpenAI

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_AZURE` | Enable Azure OpenAI models | Boolean | `true`, `false` |
| `AZURE_API_KEY` | Azure deployment API key | String | `sk-1234567890` |
| `AZURE_DEPLOYMENT` | Azure OpenAI Deployment Name | String | `skyvern-deployment` |
| `AZURE_API_BASE` | Azure deployment API base URL | String | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version | String | `2024-02-01` |

Recommended `LLM_KEY`: `AZURE_OPENAI`

### AWS Bedrock

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_BEDROCK` | Enable AWS Bedrock models.  Configure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) first. | Boolean | `true`, `false` |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

### Gemini

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_GEMINI` | Enable Gemini models | Boolean | `true`, `false` |
| `GEMINI_API_KEY` | Gemini API Key | String | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

### Ollama

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OLLAMA` | Enable local models via Ollama | Boolean | `true`, `false` |
| `OLLAMA_SERVER_URL` | Ollama server URL | String | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Ollama model name | String | `qwen2.5:7b-instruct` |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

### OpenRouter

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OPENROUTER` | Enable OpenRouter models | Boolean | `true`, `false` |
| `OPENROUTER_API_KEY` | OpenRouter API key | String | `sk-1234567890` |
| `OPENROUTER_MODEL` | OpenRouter model name | String | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | String | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

### OpenAI-Compatible

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OPENAI_COMPATIBLE` | Enable OpenAI-compatible API endpoint | Boolean | `true`, `false` |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name | String | `yi-34b`, `gpt-3.5-turbo`, `mistral-large` |
| `OPENAI_COMPATIBLE_API_KEY` | API key | String | `sk-1234567890` |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL | String | `https://api.together.xyz/v1`, `http://localhost:8000/v1` |
| `OPENAI_COMPATIBLE_API_VERSION` | API version (optional) | String | `2023-05-15` |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Max tokens (optional) | Integer | `4096`, `8192` |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature (optional) | Float | `0.0`, `0.5`, `0.7` |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Supports vision (optional) | Boolean | `true`, `false` |

Supported LLM Key: `OPENAI_COMPATIBLE`

### General LLM Configuration

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `LLM_KEY` | Model to use | String | See supported LLM keys |
| `SECONDARY_LLM_KEY` | Mini agent model | String | See supported LLM keys |
| `LLM_CONFIG_MAX_TOKENS` | Override max tokens | Integer | `128000` |

## Feature Roadmap

Future features:
*   [x] **Open Source**
*   [x] **Workflow support**
*   [x] **Improved context**
*   [x] **Cost Savings**
*   [x] **Self-serve UI**
*   [x] **Workflow UI Builder**
*   [x] **Chrome Viewport streaming**
*   [x] **Past Runs UI**
*   [X] **Auto workflow builder ("Observer") mode**
*   [x] **Prompt Caching**
*   [x] **Web Evaluation Dataset**
*   [ ] **Improved Debug mode**
*   [ ] **Chrome Extension**
*   [ ] **Skyvern Action Recorder**
*   [ ] **Interactable Livestream**
*   [ ] **Integrate LLM Observability tools**
*   [x] **Langchain Integration**

## Contributing

Contributions are welcome!  Please review the [contribution guide](CONTRIBUTING.md) and "Help Wanted" issues to get started:
*   [CONTRIBUTING.md](CONTRIBUTING.md)
*   ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)

Use [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme) to get a high-level understanding of Skyvern.

## Telemetry

By Default, Skyvern collects basic usage statistics.  Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern is open source and licensed under the [AGPL-3.0 License](LICENSE), with the exception of anti-bot measures available in our managed cloud offering.  Contact us at [support@skyvern.com](mailto:support@skyvern.com) if you have licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)