<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
</h1>

<p align="center">
  <strong>Automate web workflows effortlessly with AI: Skyvern empowers you to automate complex browser-based tasks using LLMs and Computer Vision.</strong>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Docs"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

[Skyvern](https://www.skyvern.com) uses the power of Large Language Models (LLMs) and computer vision to automate repetitive, time-consuming browser-based tasks. Say goodbye to brittle scripts and hello to intelligent automation that adapts to website changes.

## Key Features

*   **Intelligent Automation:** Skyvern leverages LLMs to understand and interact with websites, even those it's never seen before.
*   **Resilient to Changes:** Adapts to website layout updates, reducing maintenance headaches.
*   **Versatile:** Automate a wide range of workflows across numerous websites.
*   **Advanced Reasoning:** Handles complex situations, like inferring answers to questions based on context.
*   **Flexible Deployment:** Run Skyvern locally or in the cloud via [Skyvern Cloud](https://app.skyvern.com).

Want to see Skyvern in action? Check out the [#real-world-examples-of-skyvern](#real-world-examples-of-skyvern) section.

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

**UI (Recommended):**

Start the Skyvern service and UI

```bash
skyvern run all
```

Open your browser to http://localhost:8080 to start a task.

**Code:**

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

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

## How Skyvern Works

Skyvern's architecture is inspired by task-driven autonomous agents, using a swarm of agents to comprehend a website, and plan and execute its actions.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" />
</picture>

This allows Skyvern to:

1.  Operate on unseen websites by mapping visual elements to actions.
2.  Be resilient to website changes, using visual understanding instead of fixed selectors.
3.  Apply a single workflow to many sites by reasoning through interactions.
4.  Leverage LLMs for complex scenarios, like handling nuanced questions.

For a detailed technical report, see [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<!-- Redo demo -->
https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern boasts SOTA performance on the [WebBench benchmark](webbench.ai), achieving a 64.4% accuracy.  See the full evaluation [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

### Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern excels on WRITE tasks (e.g., filling forms, logging in, downloading files), making it ideal for Robotic Process Automation (RPA).

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Tasks Performance"/>
</p>

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

Define a `data_extraction_schema`:

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

## Docker Compose setup

1.  Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.
2.  Ensure no local Postgres instance is running (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to create a `.env` file.
5.  Populate your LLM provider key in the [docker-compose.yml](./docker-compose.yml). *If running Skyvern on a remote server, set the UI container's correct server IP in [docker-compose.yml](./docker-compose.yml).*
6.  Execute:

```bash
docker compose up -d
```

7.  Access the UI at `http://localhost:8080`.

> **Important:** Remove the original Postgres container if switching from the CLI-managed Postgres to Docker Compose:
>
> ```bash
> docker rm -f postgresql-container
> ```

## Skyvern Features

### Skyvern Tasks

Tasks are the core building blocks for each interaction with a website, taking a `url`, `prompt`, and optional `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Tasks"/>
</p>

### Skyvern Workflows

Workflows allow you to chain multiple tasks together, creating complex automated processes.

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Workflow Example"/>
</p>

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

### Livestreaming

Monitor Skyvern's actions in real-time with browser viewport livestreaming.

### Form Filling

Skyvern fills form inputs with ease using information provided in the `navigation_goal`.

### Data Extraction

Extract structured data from websites, including custom schema support.

### File Downloading

Download files directly from websites, with automatic upload to block storage.

### Authentication

Simplify automation behind logins with support for various authentication methods.

### 🔐 2FA Support (TOTP)

Automate workflows requiring 2FA, supporting:

1.  QR-based 2FA (e.g., Google Authenticator)
2.  Email-based 2FA
3.  SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

Skyvern currently supports password manager integrations with:

-   [x] Bitwarden
-   [ ] 1Password
-   [ ] LastPass

### Model Context Protocol (MCP)

Skyvern supports the Model Context Protocol (MCP) to allow you to use any LLM that supports MCP.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Connect Skyvern workflows to other apps with integrations for Zapier, Make.com, and N8N.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world examples of Skyvern

See how Skyvern is being used to automate real-world tasks:

### Invoice Downloading on many different websites

[Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo"/>
</p>

### Automate the job application process

[💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Demo"/>
</p>

### Automate materials procurement for a manufacturing company

[💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Finditparts Demo"/>
</p>

### Navigating to government websites to register accounts or fill out forms

[💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="California EDD Demo"/>
</p>

### Filling out random contact us forms

[💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Forms Demo"/>
</p>

### Retrieving insurance quotes from insurance providers in any language

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

Set up your development environment to use pre-commit (our commit hook handler):

```bash
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` in your browser to start using the UI
    *The Skyvern CLI supports Windows, WSL, macOS, and Linux environments.*

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com). Please reach out if anything is unclear or missing by opening an issue or contacting us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                                       |
| :------------ | :--------------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                      |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                 |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)         |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                  |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)             |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

### Environment Variables

##### OpenAI

| Variable            | Description              | Type    | Sample Value         |
| :------------------ | :----------------------- | :------ | :------------------- |
| `ENABLE_OPENAI`     | Register OpenAI models   | Boolean | `true`, `false`      |
| `OPENAI_API_KEY`    | OpenAI API Key           | String  | `sk-1234567890`      |
| `OPENAI_API_BASE`   | OpenAI API Base, optional | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional | String  | `your-org-id`     |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable            | Description          | Type    | Sample Value         |
| :------------------ | :------------------- | :------ | :------------------- |
| `ENABLE_ANTHROPIC`  | Register Anthropic models | Boolean | `true`, `false`      |
| `ANTHROPIC_API_KEY` | Anthropic API key    | String  | `sk-1234567890`      |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable            | Description                  | Type    | Sample Value                |
| :------------------ | :--------------------------- | :------ | :-------------------------- |
| `ENABLE_AZURE`      | Register Azure OpenAI models | Boolean | `true`, `false`             |
| `AZURE_API_KEY`     | Azure deployment API key     | String  | `sk-1234567890`             |
| `AZURE_DEPLOYMENT`  | Azure OpenAI Deployment Name | String  | `skyvern-deployment`       |
| `AZURE_API_BASE`    | Azure deployment api base url | String  | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version            | String  | `2024-02-01`                |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable         | Description                                                                                                    | Type    | Sample Value         |
| :--------------- | :------------------------------------------------------------------------------------------------------------- | :------ | :------------------- |
| `ENABLE_BEDROCK` | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false`      |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable         | Description          | Type    | Sample Value            |
| :--------------- | :------------------- | :------ | :---------------------- |
| `ENABLE_GEMINI`  | Register Gemini models | Boolean | `true`, `false`         |
| `GEMINI_API_KEY` | Gemini API Key       | String  | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable           | Description                               | Type    | Sample Value                      |
| :----------------- | :---------------------------------------- | :------ | :-------------------------------- |
| `ENABLE_OLLAMA`    | Register local models via Ollama          | Boolean | `true`, `false`                  |
| `OLLAMA_SERVER_URL`| URL for your Ollama server                | String  | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`     | Ollama model name to load                  | String  | `qwen2.5:7b-instruct`            |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable            | Description           | Type    | Sample Value         |
| :------------------ | :-------------------- | :------ | :------------------- |
| `ENABLE_OPENROUTER` | Register OpenRouter models | Boolean | `true`, `false`      |
| `OPENROUTER_API_KEY` | OpenRouter API key    | String  | `sk-1234567890`      |
| `OPENROUTER_MODEL`  | OpenRouter model name | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL    | String  | `https://api.openrouter.ai/v1`      |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                     | Description                                      | Type    | Sample Value                |
| :--------------------------- | :----------------------------------------------- | :------ | :-------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`  | Register a custom OpenAI-compatible API endpoint | Boolean | `true`, `false`             |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint        | String  | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc. |
| `OPENAI_COMPATIBLE_API_KEY`  | API key for OpenAI-compatible endpoint          | String  | `sk-1234567890`             |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint         | String  | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc. |
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional | String  | `2023-05-15`                |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional           | Integer | `4096`, `8192`, etc.          |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional                    | Float   | `0.0`, `0.5`, `0.7`, etc.     |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional                  | Boolean | `true`, `false`             |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable           | Description                               | Type    | Sample Value   |
| :----------------- | :---------------------------------------- | :------ | :------------- |
| `LLM_KEY`          | The name of the model you want to use     | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |

## Feature Roadmap

Our planned features for the coming months.

-   \[x] **Open Source** - Open Source Skyvern's core codebase
-   \[x] **Workflow support** - Allow support to chain multiple Skyvern calls together
-   \[x] **Improved context** - Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
-   \[x] **Cost Savings** - Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
-   \[x] **Self-serve UI** - Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
-   \[x] **Workflow UI Builder** - Introduce a UI to allow users to build and analyze workflows visually
-   \[x] **Chrome Viewport streaming** - Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
-   \[x] **Past Runs UI** - Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
-   \[X] **Auto workflow builder ("Observer") mode** - Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
-   \[x] **Prompt Caching** - Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
-   \[x] **Web Evaluation Dataset** - Integrate Skyvern with public benchmark tests to track the quality of our models over time
-   \[ ] **Improved Debug mode** - Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
-   \[ ] **Chrome Extension** - Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
-   \[ ] **Skyvern Action Recorder** - Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
-   \[ ] **Interactable Livestream** - Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
-   \[ ] **Integrate LLM Observability tools** - Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
-   \[x] **Langchain Integration** - Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We welcome your contributions! Please open a PR/issue or reach out [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3). Review the [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started.

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

By Default, Skyvern collects basic usage statistics to help us understand how Skyvern is being used. If you would like to opt-out of telemetry, please set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's open-source repository is licensed under the [AGPL-3.0 License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)