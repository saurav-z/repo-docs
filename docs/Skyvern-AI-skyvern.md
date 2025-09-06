<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
    </picture>
  </a>
  <br>
  <br>
  <p align="center"><b>Automate browser-based workflows effortlessly with Skyvern, leveraging the power of LLMs and computer vision.</b></p>
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Follow on Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="Follow on LinkedIn"/></a>
</p>

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo">
</p>

<p align="center">
  <a href="https://github.com/Skyvern-AI/skyvern">View the Skyvern Repository on GitHub</a>
</p>

## Key Features

*   **Automated Browser Workflows:** Automate complex tasks across various websites.
*   **LLM & Computer Vision Powered:** Leverage the latest AI to interact with web pages, overcoming website layout changes.
*   **No-Code Automation:** Eliminates the need for brittle, custom-built scripts.
*   **Workflow Capabilities:** Chain tasks, extract data, and automate multi-step processes.
*   **Form Filling & Data Extraction:** Seamlessly fill forms and extract structured data.
*   **Advanced Integrations:**  Integrations for [Zapier](https://docs.skyvern.com/integrations/zapier), [Make.com](https://docs.skyvern.com/integrations/make.com), and [N8N](https://docs.skyvern.com/integrations/n8n)
*   **Livestreaming:** Watch Skyvern in action with browser viewport streaming.
*   **Authentication Support:** Handles various authentication methods, including 2FA.
*   **Password Manager Integrations:** Integrates with Bitwarden (with more coming soon).

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

Start the Skyvern service and UI

```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern starts running the task in a browser that pops up and closes it when the task is done. You will be able to view the task from http://localhost:8080/history

You can also run a task on different targets:
```python
from skyvern import Skyvern

# Run on Skyvern Cloud
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Local Skyvern service
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

##  How Skyvern Works

Skyvern employs a task-driven autonomous agent design, similar to [BabyAGI](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT), but with browser automation via [Playwright](https://playwright.dev/).  It utilizes a swarm of agents to understand, plan, and execute actions on websites, offering advantages such as:

*   Works on unseen websites without custom code.
*   Resistant to website layout changes.
*   Applies a single workflow to many sites.
*   Leverages LLMs for complex scenarios, such as inferring information and handling variations in data.

For more details, see the detailed technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Performance & Evaluation

Skyvern achieves state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai), with a 64.4% accuracy.

*   **SOTA on WRITE Tasks:** Skyvern excels in tasks involving form filling, logins, and file downloads.
    <p align="center">
      <img src="fern/images/performance/webbench_write.png" alt="WebBench WRITE Performance">
    </p>

## Advanced Usage

### Control Your Own Browser (Chrome)

> **Warning:**  Due to changes in Chrome 136, you might need to copy your default user data directory. Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser.

1.  **With Python Code:**

```python
from skyvern import Skyvern

# The path to your Chrome browser (example for Mac).
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

2.  **With Skyvern Service:**

    Add these variables to your `.env` file:

```bash
CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BROWSER_TYPE=cdp-connect
```

    Restart the Skyvern service with `skyvern run all`.

### Run Skyvern with any Remote Browser

Get the CDP connection URL and pass it to Skyvern:

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get Consistent Output Schema

Use the `data_extraction_schema` parameter:

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

### Debugging Commands

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

1.  Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.
2.  Make sure you don't have postgres running locally (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file.
5.  Populate your LLM provider key in [docker-compose.yml](./docker-compose.yml).
6.  Run: `docker compose up -d`
7.  Access the UI at `http://localhost:8080`.

> **Important:**  Remove the existing Postgres container before running with Docker Compose: `docker rm -f postgresql-container`.

## Skyvern Features

### Skyvern Tasks

Tasks are the foundation of Skyvern, allowing users to navigate a website and accomplish a specific goal. They require a `url` and `prompt`, with optional `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Example">
</p>

### Skyvern Workflows

Workflows let you chain multiple tasks for complex automation.

Examples:

*   Downloading invoices.
*   Automating e-commerce purchases.

Supported features:

*   Navigation
*   Action
*   Data Extraction
*   Loops
*   File parsing
*   File upload
*   Email sending
*   Text Prompts
*   Tasks (general)
*   (Coming soon) Conditionals
*   (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Workflow Example">
</p>

### Livestreaming

View the browser viewport in real-time for debugging.

### Form Filling

Natively supports form input on websites.

### Data Extraction

Extract structured data from websites.

```jsonc
// Example data_extraction_schema in jsonc format
{
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
```

### File Downloading

Download files from websites with automatic block storage upload.

### Authentication

Supports authentication methods. Contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3) for assistance.

### 🔐 2FA Support (TOTP)

Supports various 2FA methods, like QR-based, email, and SMS.

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

Currently supports:

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Supports the Model Context Protocol (MCP).

See MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Integrate with [Zapier](https://docs.skyvern.com/integrations/zapier), [Make.com](https://docs.skyvern.com/integrations/make.com), and [N8N](https://docs.skyvern.com/integrations/n8n).

🔐 Learn more about integrations [here](https://docs.skyvern.com/integrations/zapier).

## Real-world examples of Skyvern

See how Skyvern is being used:

*   **Invoice Downloading**
    <p align="center">
      <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading">
    </p>
    [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)
*   **Job Application Automation**
    <p align="center">
      <img src="fern/images/job_application_demo.gif" alt="Job Application Demo">
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
*   **Materials Procurement**
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement Demo">
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
*   **Government Website Automation**
    <p align="center">
      <img src="fern/images/edd_services.gif" alt="Government Website Automation">
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
*   **Contact Us Form Filling**
    <p align="center">
      <img src="fern/images/contact_forms.gif" alt="Contact Us Form Filling">
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
*   **Insurance Quote Retrieval**
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Retrieval">
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Quote">
    </p>

## Contributor Setup

For local CLI installation, use:

```bash
pip install -e .
```

Configure your development environment to use pre-commit with:

```bash
skyvern quickstart contributors
```

1.  Open UI at `http://localhost:8080`.
    *Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Find detailed documentation on our [📕 docs page](https://docs.skyvern.com).  Contact us for help [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider     | Supported Models                                                      |
|--------------|-----------------------------------------------------------------------|
| OpenAI       | gpt4-turbo, gpt-4o, gpt-4o-mini                                          |
| Anthropic    | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                     |
| Azure OpenAI | Any GPT models (better with multimodal LLMs like azure/gpt4-o)          |
| AWS Bedrock  | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)           |
| Gemini       | Gemini 2.5 Pro and flash, Gemini 2.0                                  |
| Ollama       | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter   | Access models through [OpenRouter](https://openrouter.ai)                |
| OpenAI-compatible  | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable            | Description                        | Type      | Sample Value      |
|---------------------|------------------------------------|-----------|-------------------|
| `ENABLE_OPENAI`     | Register OpenAI models             | Boolean   | `true`, `false`    |
| `OPENAI_API_KEY`    | OpenAI API Key                     | String    | `sk-1234567890`   |
| `OPENAI_API_BASE`   | OpenAI API Base (optional)         | String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID (optional) | String    | `your-org-id`     |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable            | Description                     | Type      | Sample Value      |
|---------------------|---------------------------------|-----------|-------------------|
| `ENABLE_ANTHROPIC`  | Register Anthropic models        | Boolean   | `true`, `false`    |
| `ANTHROPIC_API_KEY` | Anthropic API key                | String    | `sk-1234567890`   |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable            | Description                    | Type      | Sample Value            |
|---------------------|--------------------------------|-----------|-------------------------|
| `ENABLE_AZURE`      | Register Azure OpenAI models    | Boolean   | `true`, `false`          |
| `AZURE_API_KEY`     | Azure deployment API key       | String    | `sk-1234567890`         |
| `AZURE_DEPLOYMENT`  | Azure OpenAI Deployment Name  | String    | `skyvern-deployment`    |
| `AZURE_API_BASE`    | Azure deployment API base URL  | String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version              | String    | `2024-02-01`            |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable            | Description                                                                                                                                                               | Type      | Sample Value      |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|-------------------|
| `ENABLE_BEDROCK`      | Register AWS Bedrock models. To use AWS Bedrock, ensure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are correctly set up. | Boolean   | `true`, `false`    |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable            | Description                 | Type      | Sample Value           |
|---------------------|-----------------------------|-----------|------------------------|
| `ENABLE_GEMINI`     | Register Gemini models      | Boolean   | `true`, `false`         |
| `GEMINI_API_KEY`    | Gemini API Key              | String    | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable            | Description                      | Type      | Sample Value              |
|---------------------|----------------------------------|-----------|---------------------------|
| `ENABLE_OLLAMA`     | Register local models via Ollama | Boolean   | `true`, `false`           |
| `OLLAMA_SERVER_URL` | Ollama server URL              | String    | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`      | Ollama model name              | String    | `qwen2.5:7b-instruct`     |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable            | Description                 | Type      | Sample Value             |
|---------------------|-----------------------------|-----------|--------------------------|
| `ENABLE_OPENROUTER` | Register OpenRouter models  | Boolean   | `true`, `false`          |
| `OPENROUTER_API_KEY`| OpenRouter API key         | String    | `sk-1234567890`          |
| `OPENROUTER_MODEL`  | OpenRouter model name       | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE`| OpenRouter API base URL   | String    | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                 | Description                                     | Type      | Sample Value                       |
|--------------------------|-------------------------------------------------|-----------|------------------------------------|
| `ENABLE_OPENAI_COMPATIBLE`| Register a custom OpenAI-compatible API endpoint | Boolean   | `true`, `false`                    |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint           | String    | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY` | API key for OpenAI-compatible endpoint                | String    | `sk-1234567890`                    |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint             | String    | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional      | String    | `2023-05-15`                       |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional                    | Integer   | `4096`, `8192`, etc.             |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional                          | Float     | `0.0`, `0.5`, `0.7`, etc.        |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional                           | Boolean | `true`, `false`                  |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable            | Description                          | Type      | Sample Value |
|---------------------|--------------------------------------|-----------|--------------|
| `LLM_KEY`           | Model name to use                    | String    | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | Model name for mini agents           | String    | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM  | Integer   | `128000`     |

## Feature Roadmap

Planned features:

-   [x] **Open Source**
-   [x] **Workflow Support**
-   [x] **Improved Context**
-   [x] **Cost Savings**
-   [x] **Self-serve UI**
-   [x] **Workflow UI Builder**
-   [x] **Chrome Viewport Streaming**
-   [x] **Past Runs UI**
-   [X] **Auto Workflow Builder**
-   [x] **Prompt Caching**
-   [x] **Web Evaluation Dataset**
-   [ ] **Improved Debug Mode**
-   [ ] **Chrome Extension**
-   [ ] **Skyvern Action Recorder**
-   [ ] **Interactable Livestream**
-   [ ] **Integrate LLM Observability tools**
-   [x] **Langchain Integration**

## Contributing

Contributions are welcome! Open a PR/issue or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics.  Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).

If you have any questions around licensing, please [contact us](mailto:support@skyvern.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)