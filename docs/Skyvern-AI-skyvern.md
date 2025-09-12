<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
</h1>

<p align="center">
  <strong>Automate any browser-based workflow using LLMs and Computer Vision.</strong>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

[Skyvern](https://github.com/Skyvern-AI/skyvern) empowers you to automate complex browser interactions with the power of Large Language Models (LLMs) and Computer Vision, eliminating the need for brittle, website-specific code.

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern in Action"/>
</p>

## Key Features

*   **Intelligent Automation:**  Uses Vision LLMs to understand and interact with websites without relying on hardcoded selectors, adapting to website changes.
*   **Cross-Website Compatibility:**  Apply a single workflow across numerous websites, automating tasks with generalized reasoning.
*   **Workflow Automation:** Chain multiple tasks, including navigation, data extraction, and file handling.
*   **Real-time Livestreaming:**  Monitor browser interactions for debugging and control.
*   **Form Filling and Data Extraction:** Seamlessly fills forms and extracts structured data using schemas.
*   **Authentication Support:** Integrates with various authentication methods, including 2FA and password managers.
*   **Model Context Protocol (MCP) Support:** Utilize a wide range of LLMs that support the Model Context Protocol.
*   **Integration Ready:** Connect to your favorite services with integrations for Zapier, Make.com, and N8N.

## Quickstart

### 1. Installation

```bash
pip install skyvern
```

### 2. Run Skyvern

```bash
skyvern quickstart
```

### 3. Run a Task (UI or Code)

#### UI (Recommended)

```bash
skyvern run all
```

Visit [http://localhost:8080](http://localhost:8080) to use the UI.

#### Code

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

See the [Full Documentation](https://docs.skyvern.com/) for advanced configuration.

## Advanced Usage

### Control Your Own Browser (Chrome)

> ⚠️  Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1.  **With Python Code**

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

2.  **With Skyvern Service**

Add two variables to your .env file:

```bash
# The path to your Chrome browser. This example path is for Mac.
CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BROWSER_TYPE=cdp-connect
```

Restart Skyvern service `skyvern run all` and run the task through UI or code.

### Run Skyvern with Any Remote Browser

Grab the CDP connection URL and pass it to Skyvern.

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get Consistent Output Schema from Your Run

You can do this by adding the `data_extraction_schema` parameter.

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

### Helpful Commands to Debug Issues

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

1.  Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine.
2.  Make sure you don't have postgres running locally.  (Run `docker ps` to check)
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file. This will be copied into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml).  *If you want to run Skyvern on a remote server, make sure you set the correct server IP for the UI container in [docker-compose.yml](./docker-compose.yml).*
6.  Run the following command via the commandline:
    ```bash
     docker compose up -d
    ```
7.  Navigate to `http://localhost:8080` in your browser to start using the UI.

> **Important:** Only one Postgres container can run on port 5432 at a time. If you switch from the CLI-managed Postgres to Docker Compose, you must first remove the original container:
> ```bash
> docker rm -f postgresql-container
> ```

If you encounter any database related errors while using Docker to run Skyvern, check which Postgres container is running with `docker ps`.

## Skyvern Features

*   **[Tasks](#skyvern-tasks):** The core building block; execute a single request.
*   **[Workflows](#skyvern-workflows):** Chain multiple tasks for complex automation, including navigation, data extraction, loops, and file handling.
*   **[Livestreaming](#livestreaming):** Watch your automation in action.
*   **[Form Filling](#form-filling):** Natively fill out forms.
*   **[Data Extraction](#data-extraction):** Extract data with structured schema output.
*   **[File Downloading](#file-downloading):** Automatically download files.
*   **[Authentication](#authentication):** Automate tasks behind login.
    *   🔐 **2FA Support (TOTP):** Google Authenticator, Authy, Email and SMS based 2FA
    *   🔐 **Password Manager Integrations:** Bitwarden

## Real-world examples of Skyvern

Discover how Skyvern is automating web tasks:

*   **Invoice Downloading** ([Book a demo](https://meetings.hubspot.com/skyvern/demo))
    <p align="center">
      <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading"/>
    </p>
*   **Job Application Automation** ([See it in action](https://app.skyvern.com/tasks/create/job_application))
    <p align="center">
      <img src="fern/images/job_application_demo.gif" alt="Job Application Automation"/>
    </p>
*   **Materials Procurement** ([See it in action](https://app.skyvern.com/tasks/create/finditparts))
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement"/>
    </p>
*   **Government Website Automation** ([See it in action](https://app.skyvern.com/tasks/create/california_edd))
    <p align="center">
      <img src="fern/images/edd_services.gif" alt="Government Website Automation"/>
    </p>
*   **Contact Us Form Filling** ([See it in action](https://app.skyvern.com/tasks/create/contact_us_forms))
    <p align="center">
      <img src="fern/images/contact_forms.gif" alt="Contact Us Form Filling"/>
    </p>
*   **Insurance Quote Retrieval**
    ([See it in action](https://app.skyvern.com/tasks/create/bci_seguros))
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Retrieval"/>
    </p>
    [See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Insurance Quote Retrieval"/>
    </p>

## Documentation

Detailed documentation can be found on our [documentation page](https://docs.skyvern.com/).

## Supported LLMs

| Provider       | Supported Models                                                                  |
| -------------- | --------------------------------------------------------------------------------- |
| OpenAI         | gpt4-turbo, gpt-4o, gpt-4o-mini                                                    |
| Anthropic      | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                              |
| Azure OpenAI   | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)           |
| AWS Bedrock    | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                      |
| Gemini         | Gemini 2.5 Pro and flash, Gemini 2.0                                             |
| Ollama         | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)     |
| OpenRouter     | Access models through [OpenRouter](https://openrouter.ai)                         |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable            | Description                                   | Type    | Sample Value         |
| ------------------- | --------------------------------------------- | ------- | -------------------- |
| `ENABLE_OPENAI`     | Register OpenAI models                       | Boolean | `true`, `false`      |
| `OPENAI_API_KEY`    | OpenAI API Key                                | String  | `sk-1234567890`      |
| `OPENAI_API_BASE`   | OpenAI API Base, optional                   | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional          | String  | `your-org-id`        |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable             | Description                      | Type    | Sample Value        |
| -------------------- | -------------------------------- | ------- | ------------------- |
| `ENABLE_ANTHROPIC`   | Register Anthropic models        | Boolean | `true`, `false`     |
| `ANTHROPIC_API_KEY`  | Anthropic API key                | String  | `sk-1234567890`     |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable              | Description                         | Type    | Sample Value                        |
| --------------------- | ----------------------------------- | ------- | ----------------------------------- |
| `ENABLE_AZURE`        | Register Azure OpenAI models        | Boolean | `true`, `false`                     |
| `AZURE_API_KEY`       | Azure deployment API key            | String  | `sk-1234567890`                     |
| `AZURE_DEPLOYMENT`    | Azure OpenAI Deployment Name        | String  | `skyvern-deployment`                |
| `AZURE_API_BASE`      | Azure deployment api base url       | String  | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`   | Azure API Version                   | String  | `2024-02-01`                        |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable             | Description                                                                                                                                          | Type    | Sample Value |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------ |
| `ENABLE_BEDROCK`     | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false` |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable          | Description               | Type    | Sample Value                   |
| ----------------- | ------------------------- | ------- | ------------------------------ |
| `ENABLE_GEMINI`   | Register Gemini models    | Boolean | `true`, `false`                |
| `GEMINI_API_KEY`  | Gemini API Key            | String  | `your_google_gemini_api_key`   |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable            | Description                      | Type    | Sample Value                         |
| ------------------- | -------------------------------- | ------- | ------------------------------------ |
| `ENABLE_OLLAMA`     | Register local models via Ollama | Boolean | `true`, `false`                      |
| `OLLAMA_SERVER_URL` | URL for your Ollama server       | String  | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`      | Ollama model name to load      | String  | `qwen2.5:7b-instruct`                |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable              | Description                      | Type    | Sample Value                  |
| --------------------- | -------------------------------- | ------- | ----------------------------- |
| `ENABLE_OPENROUTER`   | Register OpenRouter models       | Boolean | `true`, `false`               |
| `OPENROUTER_API_KEY`  | OpenRouter API key               | String  | `sk-1234567890`              |
| `OPENROUTER_MODEL`    | OpenRouter model name            | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL          | String  | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                         | Description                                     | Type    | Sample Value                      |
| -------------------------------- | ----------------------------------------------- | ------- | --------------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`       | Register a custom OpenAI-compatible API endpoint | Boolean | `true`, `false`                   |
| `OPENAI_COMPATIBLE_MODEL_NAME`  | Model name for OpenAI-compatible endpoint        | String  | `yi-34b`, `gpt-3.5-turbo`, etc.   |
| `OPENAI_COMPATIBLE_API_KEY`     | API key for OpenAI-compatible endpoint            | String  | `sk-1234567890`                   |
| `OPENAI_COMPATIBLE_API_BASE`    | Base URL for OpenAI-compatible endpoint           | String  | `https://api.together.xyz/v1`, `http://localhost:8000/v1` |
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional | String  | `2023-05-15`                      |
| `OPENAI_COMPATIBLE_MAX_TOKENS`  | Maximum tokens for completion, optional          | Integer | `4096`, `8192`, etc.              |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional                   | Float   | `0.0`, `0.5`, `0.7`, etc.           |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION`| Whether model supports vision, optional         | Boolean   | `true`, `false`           |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable               | Description                         | Type    | Sample Value |
| ---------------------- | ----------------------------------- | ------- | ------------ |
| `LLM_KEY`              | The name of the model you want to use | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY`    | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000`     |

## Feature Roadmap

*   [x] Open Source
*   [x] Workflow Support
*   [x] Improved Context
*   [x] Cost Savings
*   [x] Self-serve UI
*   [x] Workflow UI Builder
*   [x] Chrome Viewport streaming
*   [x] Past Runs UI
*   [X] Auto workflow builder ("Observer") mode
*   [x] Prompt Caching
*   [x] Web Evaluation Dataset
*   [ ] Improved Debug mode
*   [ ] Chrome Extension
*   [ ] Skyvern Action Recorder
*   [ ] Interactable Livestream
*   [ ] Integrate LLM Observability tools
*   [x] Langchain Integration

## Contributing

We welcome contributions! See the [contribution guide](CONTRIBUTING.md) and the ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for opportunities.  For general questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects usage statistics; opt-out by setting `SKYVERN_TELEMETRY=false`.

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).  Commercial offerings available via a managed cloud.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)