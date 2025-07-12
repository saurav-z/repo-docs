<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br/>
  Skyvern: Automate Browser Workflows with LLMs and Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

**Skyvern empowers you to automate complex browser-based tasks using the power of Large Language Models (LLMs) and computer vision, revolutionizing web automation.**  Visit the [Skyvern GitHub repository](https://github.com/Skyvern-AI/skyvern) to get started.

## Key Features

*   **Automated Browser Interaction:**  Control browsers programmatically to interact with any website.
*   **LLM-Driven Automation:** Leverage LLMs to understand and execute complex workflows, even on unfamiliar sites.
*   **Resilient to Website Changes:** Adapt to website layout updates without requiring code modifications.
*   **Versatile Workflow Execution:**  Apply a single workflow across multiple websites.
*   **Form Filling & Data Extraction:**  Automate form completion and extract structured data.
*   **File Downloading & Uploading:**  Download files and seamlessly upload them to block storage.
*   **Real-time Livestreaming:** Observe Skyvern's actions in real-time for debugging and control.
*   **2FA & Password Manager Support:**  Securely automate tasks behind login screens with 2FA and password manager integrations.
*   **Integrations:** Easily integrate with Zapier, Make.com, and N8N for extended functionality.

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

## How It Works

Skyvern utilizes a multi-agent system inspired by task-driven autonomous agents. It interacts with websites using browser automation, giving it the ability to map visual elements to actions, and reason through the steps needed to complete a task.  This approach offers several advantages:

*   **Adaptability:** Operates on websites it hasn't seen before.
*   **Resilience:**  Unaffected by website layout changes.
*   **Scalability:**  Applies a single workflow to numerous websites.
*   **Intelligent Reasoning:**  Leverages LLMs to handle complex scenarios (e.g., understanding context, interpreting data variations).

For detailed technical insights, explore the technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

[Demo Video](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

## Performance & Evaluation

Skyvern demonstrates state-of-the-art performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.  See the detailed report [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

## Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

## Advanced Usage

### Control your own browser (Chrome)
> ⚠️ WARNING: Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1. Just With Python Code
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

2. With Skyvern Service

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

## Docker Compose Setup

1.  Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
2.  Ensure you don't have Postgres running locally (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file.  This will be copied into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml). *If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).*
6.  Run `docker compose up -d`.
7.  Access the UI at `http://localhost:8080`.

>   **Important:** Only one Postgres container can run on port 5432. If switching from the CLI-managed Postgres to Docker Compose, first remove the original container: `docker rm -f postgresql-container`

If you encounter database errors with Docker, check which Postgres container is running using `docker ps`.

## Skyvern Features Details

### Skyvern Tasks

Tasks form the core of Skyvern, representing a single request to navigate and interact with a website to achieve a specific goal. Tasks require a `url` and `prompt`, and can optionally include a `data_extraction_schema` for structured output and `error codes` for handling specific situations.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows

Workflows enable you to chain multiple tasks to create automated work units.

For example, download invoices, automate job applications, or automate e-commerce purchases

Supported workflow features:

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

Skyvern's livestreaming feature lets you monitor the browser viewport in real-time, facilitating debugging and intervention when needed.

### Form Filling

Skyvern is built with form filling capabilities. The `navigation_goal` is used to provide Skyvern with the information needed to fill out form inputs.

### Data Extraction

Skyvern offers data extraction capabilities.

You can specify a `data_extraction_schema` within the prompt in JSONC format to structure Skyvern's output.

### File Downloading

Skyvern enables file downloads, with automatic uploads to block storage (if configured), accessible via the UI.

### Authentication

Skyvern supports multiple authentication methods, including:

#### 🔐 2FA Support (TOTP)

Skyvern allows you to automate workflows that require 2FA.

Examples include:

1.  QR-based 2FA (e.g. Google Authenticator, Authy)
2.  Email based 2FA
3.  SMS based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

#### Password Manager Integrations

Skyvern supports integrations with password managers:

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Skyvern supports the Model Context Protocol (MCP) to allow you to use any LLM that supports MCP.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Skyvern supports Zapier, Make.com, and N8N to allow you to connect your Skyvern workflows to other apps.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world Examples

See Skyvern in action with real-world use cases. Contribute your own examples via PRs!

*   **Invoice Downloading:** Automate invoice downloads across various websites. ([Book a demo](https://meetings.hubspot.com/skyvern/demo))
    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   **Job Application Automation:**  Streamline the job application process.  [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>
*   **Materials Procurement:** Automate materials procurement for manufacturing.  [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>
*   **Government Website Navigation:**  Register accounts or fill out forms on government websites.  [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>
    <!-- Add example of delaware entity lookups x2 -->
*   **Contact Form Filling:** Automate filling of contact us forms. [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>
*   **Insurance Quote Retrieval:** Retrieve insurance quotes from providers in any language. [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Contributor Setup

For complete CLI installation in a local environment:

```bash
pip install -e .
```

Set up your development environment to use pre-commit:

```bash
skyvern quickstart contributors
```

1.  Access the UI at `http://localhost:8080`.
    *The Skyvern CLI supports Windows, WSL, macOS, and Linux environments.*

## Documentation

Comprehensive documentation is available on our [📕 docs page](https://docs.skyvern.com). For questions or feedback, please open an issue or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider     | Supported Models                                                                                                  |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| OpenAI       | gpt4-turbo, gpt-4o, gpt-4o-mini                                                                                 |
| Anthropic    | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                             |
| Azure OpenAI | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                         |
| AWS Bedrock  | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                     |
| Gemini       | Gemini 2.5 Pro and flash, Gemini 2.0                                                                              |
| Ollama       | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)                                         |
| OpenRouter   | Access models through [OpenRouter](https://openrouter.ai)                                                         |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable          | Description                     | Type    | Sample Value        |
| ----------------- | ------------------------------- | ------- | ------------------- |
| `ENABLE_OPENAI`   | Register OpenAI models          | Boolean | `true`, `false`     |
| `OPENAI_API_KEY`  | OpenAI API Key                  | String  | `sk-1234567890`     |
| `OPENAI_API_BASE` | OpenAI API Base, optional       | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional | String | `your-org-id`     |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable           | Description                    | Type    | Sample Value        |
| ------------------ | ------------------------------ | ------- | ------------------- |
| `ENABLE_ANTHROPIC` | Register Anthropic models     | Boolean | `true`, `false`     |
| `ANTHROPIC_API_KEY` | Anthropic API key            | String  | `sk-1234567890`     |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable           | Description                         | Type    | Sample Value        |
| ------------------ | ----------------------------------- | ------- | ------------------- |
| `ENABLE_AZURE`     | Register Azure OpenAI models        | Boolean | `true`, `false`     |
| `AZURE_API_KEY`    | Azure deployment API key            | String  | `sk-1234567890`     |
| `AZURE_DEPLOYMENT` | Azure OpenAI Deployment Name       | String  | `skyvern-deployment`|
| `AZURE_API_BASE`   | Azure deployment api base url     | String  | `https://skyvern-deployment.openai.azure.com/`|
| `AZURE_API_VERSION`| Azure API Version                 | String  | `2024-02-01`         |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable           | Description                                                                                                                                               | Type    | Sample Value        |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------- |
| `ENABLE_BEDROCK`   | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false`     |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable        | Description               | Type    | Sample Value        |
| --------------- | ------------------------- | ------- | ------------------- |
| `ENABLE_GEMINI` | Register Gemini models  | Boolean | `true`, `false`     |
| `GEMINI_API_KEY` | Gemini API Key            | String  | `your_google_gemini_api_key`|

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable           | Description                     | Type    | Sample Value                      |
| ------------------ | ------------------------------- | ------- | --------------------------------- |
| `ENABLE_OLLAMA`    | Register local models via Ollama | Boolean | `true`, `false`                 |
| `OLLAMA_SERVER_URL` | URL for your Ollama server     | String  | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`     | Ollama model name to load      | String  | `qwen2.5:7b-instruct`             |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable           | Description                  | Type    | Sample Value        |
| ------------------ | ---------------------------- | ------- | ------------------- |
| `ENABLE_OPENROUTER` | Register OpenRouter models | Boolean | `true`, `false`     |
| `OPENROUTER_API_KEY` | OpenRouter API key          | String  | `sk-1234567890`     |
| `OPENROUTER_MODEL` | OpenRouter model name        | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL  | String  | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                     | Description                                            | Type    | Sample Value        |
| ---------------------------- | ------------------------------------------------------ | ------- | ------------------- |
| `ENABLE_OPENAI_COMPATIBLE`   | Register a custom OpenAI-compatible API endpoint     | Boolean | `true`, `false`     |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint          | String  | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY` | API key for OpenAI-compatible endpoint              | String  | `sk-1234567890`     |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint            | String  | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional | String | `2023-05-15`         |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional               | Integer | `4096`, `8192`, etc.  |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional                      | Float   | `0.0`, `0.5`, `0.7`, etc.  |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional              | Boolean | `true`, `false`     |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable            | Description                                  | Type    | Sample Value        |
| ------------------- | -------------------------------------------- | ------- | ------------------- |
| `LLM_KEY`           | The name of the model you want to use       | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | The name of the model for mini agents skyvern runs with  | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |

## Feature Roadmap

Planned features for the coming months. Share your suggestions via [email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

-   \[x] Open Source
-   \[x] Workflow support
-   \[x] Improved context
-   \[x] Cost Savings
-   \[x] Self-serve UI
-   \[x] Workflow UI Builder
-   \[x] Chrome Viewport streaming
-   \[x] Past Runs UI
-   \[x] Auto workflow builder ("Observer") mode
-   \[x] Prompt Caching
-   \[x] Web Evaluation Dataset
-   \[ ] Improved Debug mode
-   \[ ] Chrome Extension
-   \[ ] Skyvern Action Recorder
-   \[ ] Interactable Livestream
-   \[ ] Integrate LLM Observability tools
-   \[x] Langchain Integration

## Contributing

We welcome contributions! Open PRs/issues or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3). Review our [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

By Default, Skyvern collects basic usage statistics to help us understand how Skyvern is being used. If you would like to opt-out of telemetry, please set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's core codebase is licensed under the [AGPL-3.0 License](LICENSE).