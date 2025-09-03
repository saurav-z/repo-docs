<h1 align="center">
  Skyvern: Automate Browser Workflows with the Power of AI
</h1>

<p align="center">
  <a href="https://www.skyvern.com/">
    <img src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo" height="120">
  </a>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

**Skyvern empowers you to automate complex browser-based tasks using the intelligence of Large Language Models (LLMs) and computer vision, eliminating the need for brittle and website-specific automation scripts.**

---

## Key Features

*   **Intelligent Automation:** Automates workflows by understanding and interacting with websites using LLMs and computer vision.
*   **Resilient to Website Changes:** Works even when website layouts evolve, reducing maintenance.
*   **Cross-Website Compatibility:** Apply a single workflow across multiple websites.
*   **Advanced Reasoning:** Leverages LLMs for complex tasks, like inferring information and handling variations in data.
*   **Cloud and Local Deployment:** Supports both cloud (Skyvern Cloud) and local deployment options.
*   **Workflows:** Chain multiple tasks together for complex automations, with features like navigation, data extraction, and loops.
*   **Livestreaming:** View Skyvern's actions in real-time for debugging and understanding.
*   **Form Filling:** Seamlessly fills out forms on websites.
*   **Data Extraction:** Extracts structured data from websites based on your specifications.
*   **File Downloading:** Downloads files and uploads them to block storage (if configured).
*   **Authentication Support:** Offers authentication methods, including 2FA, and password manager integrations.
*   **Integration Options:** Supports Zapier, Make.com, and N8N for connecting to other applications.
*   **Model Context Protocol (MCP):** Allows you to use any LLM that supports MCP.

---

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

Start the Skyvern service and UI:

```bash
skyvern run all
```

Go to [http://localhost:8080](http://localhost:8080) and use the UI to run a task.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern will run the task in a browser that pops up and closes when finished. You can view the task from [http://localhost:8080/history](http://localhost:8080/history).

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

---

## How Skyvern Works

Inspired by task-driven autonomous agents like BabyAGI and AutoGPT, Skyvern adds the ability to interact with websites using browser automation libraries like [Playwright](https://playwright.dev/).

Skyvern uses a swarm of agents to comprehend a website, and plan and execute its actions:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" />
</picture>

This enables Skyvern to:

*   Operate on websites it's never seen before.
*   Adapt to website layout changes.
*   Apply a single workflow to many websites.
*   Handle complex situations using LLMs, like inferring information.

For more technical details, see the detailed report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

---

## Demo

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Demo of Skyvern in action">
</p>

---

## Performance and Evaluation

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. Read the technical report and evaluation [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

<p align="center">
  <img src="fern/images/performance/webbench_overall.png"/>
</p>

### Performance on WRITE Tasks

Skyvern excels in WRITE tasks, essential for RPA (Robotic Process Automation).

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

---

## Advanced Usage

### Control your own browser (Chrome)

> ⚠️ WARNING: Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1.  Just With Python Code

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

2.  With Skyvern Service

Add two variables to your .env file:

```bash
# The path to your Chrome browser. This example path is for Mac.
CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BROWSER_TYPE=cdp-connect
```

Restart Skyvern service `skyvern run all` and run the task through UI or code

### Run Skyvern with any remote browser

Grab the cdp connection url and pass it to Skyvern

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get consistent output schema from your run

You can do this by adding the `data_extraction_schema` parameter:

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

---

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

---

## Skyvern Features in Detail

### Skyvern Tasks

Tasks are the core unit within Skyvern. Each task is a single request to Skyvern, instructing it to navigate and accomplish a specific goal.

Tasks require a `url`, `prompt`, and can optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows

Workflows are used to chain multiple tasks for cohesive work.

For example, to download invoices newer than January 1st, you could create a workflow that navigates to the invoices page, filters invoices, extracts a list of eligible invoices, and iterates to download each one.

Another example is automating e-commerce purchases, which can include navigating to the product page, adding it to a cart, validating the cart, and completing the checkout.

Supported workflow features include:

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
  <img src="fern/images/invoice_downloading_workflow_example.png"/>
</p>

### Livestreaming

Skyvern lets you livestream the browser's viewport for debugging.

### Form Filling

Skyvern is natively capable of filling out form inputs on websites.

### Data Extraction

Skyvern is also capable of extracting data from a website.

You can also specify a `data_extraction_schema` directly within the main prompt to tell Skyvern exactly what data you'd like to extract from the website, in jsonc format. Skyvern's output will be structured in accordance to the supplied schema.

### File Downloading

Skyvern is also capable of downloading files from a website. All downloaded files are automatically uploaded to block storage (if configured), and you can access them via the UI.

### Authentication

Skyvern supports a number of different authentication methods to make it easier to automate tasks behind a login. If you'd like to try it out, please reach out to us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

<p align="center">
  <img src="fern/images/secure_password_task_example.png"/>
</p>

#### 🔐 2FA Support (TOTP)

Skyvern supports several 2FA methods:

1.  QR-based 2FA (e.g., Google Authenticator, Authy)
2.  Email-based 2FA
3.  SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

#### Password Manager Integrations

Skyvern currently supports:

*   \[x] Bitwarden
*   \[ ] 1Password
*   \[ ] LastPass

### Model Context Protocol (MCP)

Skyvern supports MCP, allowing you to use any LLM supporting it.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Skyvern integrates with Zapier, Make.com, and N8N to connect to other apps.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

---

## Real-World Examples

Explore how Skyvern is being used:

*   [Invoice Downloading](https://meetings.hubspot.com/skyvern/demo)
    <p align="center">
      <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading">
    </p>
*   [Job Application Automation](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif" alt="Job Application Automation">
    </p>
*   [Materials Procurement](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement">
    </p>
*   [Government Website Navigation](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif" alt="Government Website Navigation">
    </p>
*   [Contact Form Filling](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif" alt="Contact Form Filling">
    </p>
*   [Insurance Quote Retrieval](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Retrieval">
    </p>
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Insurance Quote Retrieval">
    </p>

---

## Contributor Setup

For a complete local environment CLI Installation:

```bash
pip install -e .
```

Set up your development environment for pre-commit:

```bash
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` in your browser to start using the UI.
    *The Skyvern CLI supports Windows, WSL, macOS, and Linux.*

---

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com).  Contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3) if anything is unclear or missing.

---

## Supported LLMs

| Provider      | Supported Models                                                        |
| ------------- | ----------------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                           |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                       |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)             |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                     |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)        |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable            | Description                      | Type    | Sample Value         |
| ------------------- | -------------------------------- | ------- | -------------------- |
| `ENABLE_OPENAI`     | Register OpenAI models           | Boolean | `true`, `false`      |
| `OPENAI_API_KEY`    | OpenAI API Key                   | String  | `sk-1234567890`      |
| `OPENAI_API_BASE`   | OpenAI API Base, optional        | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional | String | `your-org-id`        |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable            | Description              | Type    | Sample Value         |
| ------------------- | ------------------------ | ------- | -------------------- |
| `ENABLE_ANTHROPIC`  | Register Anthropic models | Boolean | `true`, `false`      |
| `ANTHROPIC_API_KEY` | Anthropic API key        | String  | `sk-1234567890`      |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable            | Description                    | Type    | Sample Value                             |
| ------------------- | ------------------------------ | ------- | ---------------------------------------- |
| `ENABLE_AZURE`      | Register Azure OpenAI models   | Boolean | `true`, `false`                          |
| `AZURE_API_KEY`     | Azure deployment API key       | String  | `sk-1234567890`                          |
| `AZURE_DEPLOYMENT`  | Azure OpenAI Deployment Name | String  | `skyvern-deployment`                     |
| `AZURE_API_BASE`    | Azure deployment api base url  | String  | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version              | String  | `2024-02-01`                             |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable            | Description                                                            | Type    | Sample Value         |
| ------------------- | ---------------------------------------------------------------------- | ------- | -------------------- |
| `ENABLE_BEDROCK`    | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false`      |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable         | Description           | Type    | Sample Value            |
| ---------------- | --------------------- | ------- | ----------------------- |
| `ENABLE_GEMINI`  | Register Gemini models | Boolean | `true`, `false`         |
| `GEMINI_API_KEY` | Gemini API Key        | String  | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable            | Description                      | Type    | Sample Value                        |
| ------------------- | -------------------------------- | ------- | ----------------------------------- |
| `ENABLE_OLLAMA`     | Register local models via Ollama | Boolean | `true`, `false`                     |
| `OLLAMA_SERVER_URL` | URL for your Ollama server       | String  | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`      | Ollama model name to load       | String  | `qwen2.5:7b-instruct`              |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable            | Description                      | Type    | Sample Value            |
| ------------------- | -------------------------------- | ------- | ----------------------- |
| `ENABLE_OPENROUTER` | Register OpenRouter models       | Boolean | `true`, `false`         |
| `OPENROUTER_API_KEY`| OpenRouter API key               | String  | `sk-1234567890`         |
| `OPENROUTER_MODEL`  | OpenRouter model name            | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE`| OpenRouter API base URL          | String  | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                     | Description                                    | Type    | Sample Value                                    |
| ---------------------------- | ---------------------------------------------- | ------- | ----------------------------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`   | Register a custom OpenAI-compatible endpoint   | Boolean | `true`, `false`                                 |
| `OPENAI_COMPATIBLE_MODEL_NAME`| Model name for OpenAI-compatible endpoint      | String  | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY`  | API key for OpenAI-compatible endpoint         | String  | `sk-1234567890`                                 |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint        | String  | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional | String | `2023-05-15`                                    |
| `OPENAI_COMPATIBLE_MAX_TOKENS`| Maximum tokens for completion, optional        | Integer | `4096`, `8192`, etc.                            |
| `OPENAI_COMPATIBLE_TEMPERATURE`| Temperature setting, optional                  | Float   | `0.0`, `0.5`, `0.7`, etc.                       |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional     | Boolean | `true`, `false`                                 |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable               | Description                     | Type    | Sample Value       |
| ---------------------- | ------------------------------- | ------- | ------------------ |
| `LLM_KEY`              | The name of the model you want to use | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY`  | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS`| Override the max tokens used by the LLM | Integer | `128000` |

---

## Feature Roadmap

Our planned features:

*   \[x] **Open Source**
*   \[x] **Workflow support**
*   \[x] **Improved context**
*   \[x] **Cost Savings**
*   \[x] **Self-serve UI**
*   \[x] **Workflow UI Builder**
*   \[x] **Chrome Viewport streaming**
*   \[x] **Past Runs UI**
*   \[x] **Auto workflow builder ("Observer") mode**
*   \[x] **Prompt Caching**
*   \[x] **Web Evaluation Dataset**
*   \[ ] **Improved Debug mode**
*   \[ ] **Chrome Extension**
*   \[ ] **Skyvern Action Recorder**
*   \[ ] **Interactable Livestream**
*   \[ ] **Integrate LLM Observability tools**
*   \[x] **Langchain Integration**

---

## Contributing

We welcome contributions! Open a PR/issue or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3). See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

---

## Telemetry

By default, Skyvern collects basic usage statistics.  To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

---

## License

Skyvern's core logic is available under the [AGPL-3.0 License](LICENSE). Managed cloud offerings include additional proprietary anti-bot measures.

Contact us at [support@skyvern.com](mailto:support@skyvern.com) with licensing questions.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)