<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png"/>
  </picture>
  <br />
  Skyvern: Automate your browser-based workflows using AI!
  <br />
  <a href="https://github.com/Skyvern-AI/skyvern">
    <img src="https://img.shields.io/github/stars/skyvern-ai/skyvern?style=social" alt="Stars"/>
  </a>
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

Skyvern empowers you to automate complex browser interactions by leveraging the power of Large Language Models (LLMs) and computer vision, eliminating the need for fragile, website-specific scripts.  **[Explore the Skyvern GitHub repository](https://github.com/Skyvern-AI/skyvern)**

## Key Features

*   **LLM-Powered Automation:** Automate workflows on websites you've never seen before.
*   **Resilient to Layout Changes:**  Skyvern adapts to website updates without code modifications.
*   **Cross-Website Applicability:** Apply a single workflow across numerous websites.
*   **Advanced Reasoning:** Leverages LLMs to handle complex scenarios, such as filling out forms and extracting data.
*   **Workflow Builder:** Chain multiple tasks together to form a cohesive unit of work.
*   **Data Extraction:** Extract structured data based on a user-provided schema.
*   **Livestreaming:** Real-time browser view for debugging.
*   **2FA Support:** Secure workflows with TOTP, email, and SMS-based 2FA.
*   **Integration:** Zapier, Make.com, and N8N integrations.

## How Skyvern Works

Skyvern uses a swarm of intelligent agents that combines the capabilities of Playwright and LLMs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" />
</picture>

Skyvern was inspired by the Task-Driven autonomous agent design popularized by [BabyAGI](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT).

## Performance and Evaluation

Skyvern has SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. The technical report + evaluation can be found [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

<p align="center">
  <img src="fern/images/performance/webbench_overall.png"/>
</p>

### Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

## Quickstart

### Installation

```bash
pip install skyvern
```

### Run

```bash
skyvern quickstart
```

### Running Tasks

#### UI (Recommended)

Start the Skyvern service and UI (when DB is up and running)

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

## Docker Compose setup

1.  Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine
1.  Make sure you don't have postgres running locally (Run `docker ps` to check)
1.  Clone the repository and navigate to the root directory
1.  Run `skyvern init llm` to generate a `.env` file. This will be copied into the Docker image.
1.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml). *If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).*
1.  Run the following command via the commandline:

```bash
docker compose up -d
```

1.  Navigate to `http://localhost:8080` in your browser to start using the UI

> **Important:** Only one Postgres container can run on port 5432 at a time. If you switch from the CLI-managed Postgres to Docker Compose, you must first remove the original container:
>
> ```bash
> docker rm -f postgresql-container
> ```

If you encounter any database related errors while using Docker to run Skyvern, check which Postgres container is running with `docker ps`.

## Skyvern Features

### Skyvern Tasks

Tasks are the fundamental building blocks within Skyvern, each representing a single instruction to navigate and perform a specific action on a website.

Tasks require you to specify a `url`, `prompt`, and can optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows

Workflows allow you to chain multiple tasks together to create complex automation sequences.

Supported workflow features include:

1.  Browser Task
2.  Browser Action
3.  Data Extraction
4.  Validation
5.  For Loops
6.  File parsing
7.  Sending emails
8.  Text Prompts
9.  HTTP Request Block
10. Custom Code Block
11. Uploading files to block storage
12. (Coming soon) Conditionals

<p align="center">
  <img src="fern/images/block_example_v2.png"/>
</p>

### Authentication

Skyvern supports several authentication methods.

### 🔐 2FA Support (TOTP)

*   QR-based 2FA
*   Email-based 2FA
*   SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

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

## Real-World Examples

*   Invoice Downloading
    [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)
    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   Automate Job Applications
    [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>
*   Automate Materials Procurement
    [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>
*   Navigating Government Websites
    [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>
<!-- Add example of delaware entity lookups x2 -->
*   Filling Contact Forms
    [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>
*   Retrieving Insurance Quotes
    [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Documentation

Find detailed information on our [📕 docs page](https://docs.skyvern.com).  Please let us know if something is unclear by opening an issue or reaching out to us.

## Supported LLMs

See the table below for all the supported LLMs.

| Provider         | Supported Models                                        |
| ---------------- | ------------------------------------------------------- |
| OpenAI           | gpt4-turbo, gpt-4o, gpt-4o-mini                         |
| Anthropic        | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)     |
| Azure OpenAI     | Any GPT models. Better performance with a multimodal llm |
| AWS Bedrock      | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)   |
| Gemini           | Gemini 2.5 Pro and flash, Gemini 2.0                     |
| Ollama           | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter       | Access models through [OpenRouter](https://openrouter.ai) |
| OpenAI-compatible| Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable          | Description                       | Type    | Sample Value          |
| ----------------- | --------------------------------- | ------- | --------------------- |
| `ENABLE_OPENAI`   | Register OpenAI models            | Boolean | `true`, `false`       |
| `OPENAI_API_KEY`  | OpenAI API Key                    | String  | `sk-1234567890`       |
| `OPENAI_API_BASE` | OpenAI API Base, optional         | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional | String  | `your-org-id`       |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable             | Description                | Type    | Sample Value          |
| -------------------- | -------------------------- | ------- | --------------------- |
| `ENABLE_ANTHROPIC`   | Register Anthropic models  | Boolean | `true`, `false`       |
| `ANTHROPIC_API_KEY`  | Anthropic API key          | String  | `sk-1234567890`       |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable            | Description                        | Type    | Sample Value          |
| ------------------- | ---------------------------------- | ------- | --------------------- |
| `ENABLE_AZURE`      | Register Azure OpenAI models       | Boolean | `true`, `false`       |
| `AZURE_API_KEY`     | Azure deployment API key           | String  | `sk-1234567890`       |
| `AZURE_DEPLOYMENT`  | Azure OpenAI Deployment Name       | String  | `skyvern-deployment` |
| `AZURE_API_BASE`    | Azure deployment api base url    | String  | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version                | String  | `2024-02-01`          |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable          | Description                     | Type    | Sample Value          |
| ----------------- | ------------------------------- | ------- | --------------------- |
| `ENABLE_BEDROCK` | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false`       |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable        | Description              | Type    | Sample Value          |
| --------------- | ------------------------ | ------- | --------------------- |
| `ENABLE_GEMINI` | Register Gemini models  | Boolean | `true`, `false`       |
| `GEMINI_API_KEY` | Gemini API Key          | String  | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable           | Description                    | Type    | Sample Value              |
| ------------------ | ------------------------------ | ------- | ------------------------- |
| `ENABLE_OLLAMA`    | Register local models via Ollama | Boolean | `true`, `false`           |
| `OLLAMA_SERVER_URL`| URL for your Ollama server   | String  | `http://host.docker.internal:11434`|
| `OLLAMA_MODEL`     | Ollama model name to load    | String  | `qwen2.5:7b-instruct`    |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable            | Description                | Type    | Sample Value          |
| ------------------- | -------------------------- | ------- | --------------------- |
| `ENABLE_OPENROUTER` | Register OpenRouter models | Boolean | `true`, `false`       |
| `OPENROUTER_API_KEY`| OpenRouter API key         | String  | `sk-1234567890`       |
| `OPENROUTER_MODEL`  | OpenRouter model name      | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL  | String  | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                     | Description                               | Type    | Sample Value              |
| ---------------------------- | ----------------------------------------- | ------- | ------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`   | Register a custom OpenAI-compatible API endpoint | Boolean | `true`, `false`           |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint | String  | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY`   | API key for OpenAI-compatible endpoint      | String  | `sk-1234567890`           |
| `OPENAI_COMPATIBLE_API_BASE`  | Base URL for OpenAI-compatible endpoint     | String  | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional| String | `2023-05-15`           |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional   | Integer | `4096`, `8192`, etc.    |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional           | Float   | `0.0`, `0.5`, `0.7`, etc. |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional| Boolean | `true`, `false`           |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable             | Description                           | Type    | Sample Value         |
| -------------------- | ------------------------------------- | ------- | -------------------- |
| `LLM_KEY`            | The name of the model you want to use | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY`  | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |

## Feature Roadmap

*   **Open Source**
*   **Workflow support**
*   **Improved context**
*   **Cost Savings**
*   **Self-serve UI**
*   **Workflow UI Builder**
*   **Chrome Viewport streaming**
*   **Past Runs UI**
*   **Auto workflow builder ("Observer") mode**
*   **Prompt Caching**
*   **Web Evaluation Dataset**
*   **Improved Debug mode**
*   **Chrome Extension**
*   **Skyvern Action Recorder**
*   **Interactable Livestream**
*   **Integrate LLM Observability tools**
*   **Langchain Integration**

## Contributing

We welcome contributions! See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For high level questions check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme)

## Telemetry

By Default, Skyvern collects basic usage statistics. You can opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)