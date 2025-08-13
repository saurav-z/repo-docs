<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
    </picture>
  </a>
  <br />
  <br />
  Automate browser tasks effortlessly with Skyvern, the AI-powered web automation platform.
  <br />
  <a href="https://github.com/Skyvern-AI/skyvern">
    <img src="https://img.shields.io/github/stars/skyvern-ai/skyvern?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://discord.gg/fG2XXEuQX3">
      <img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=Discord" alt="Discord"/>
  </a>
  <a href="https://www.skyvern.com/">
      <img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/>
  </a>
  <a href="https://docs.skyvern.com/">
      <img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Docs"/>
  </a>
</h1>

[Skyvern](https://www.skyvern.com) is a powerful, AI-driven platform that automates browser-based workflows using Large Language Models (LLMs) and computer vision.  Say goodbye to brittle, code-dependent automation and hello to robust, adaptable solutions.

## Key Features

*   **AI-Powered Automation:**  Leverages LLMs and computer vision to understand and interact with websites, eliminating the need for custom scripts and brittle XPath selectors.
*   **Resilient to Website Changes:** Adapts to website layout changes, ensuring your automations remain functional over time.
*   **Cross-Website Compatibility:**  Apply single workflows across numerous websites, streamlining automation efforts.
*   **Advanced Reasoning:**  Utilizes LLMs to handle complex scenarios, such as inferring information and comparing similar products across different sites.
*   **Workflows:** Chain multiple tasks for comprehensive automation.
*   **Data Extraction:** Extract structured data from websites with schema definition support.
*   **2FA Support:**  Supports various 2FA methods for secure automation.
*   **Integration:** Integrates with Zapier, Make.com, and N8N to connect your workflows.

[Visit the original repository for more details.](https://github.com/Skyvern-AI/skyvern)

## Quickstart

### 1. Installation

```bash
pip install skyvern
```

### 2. Running Skyvern

#### Skyvern Cloud
[Skyvern Cloud](https://app.skyvern.com) is a managed cloud version of Skyvern that allows you to run Skyvern without worrying about the infrastructure. It allows you to run multiple Skyvern instances in parallel and comes bundled with anti-bot detection mechanisms, proxy network, and CAPTCHA solvers.

If you'd like to try it out, navigate to [app.skyvern.com](https://app.skyvern.com) and create an account.

#### Local Setup

Start the Skyvern service and UI

```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task

#### Code Example

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

## How It Works

Skyvern employs an innovative, agent-based architecture inspired by BabyAGI and AutoGPT. It allows agents to comprehend websites and plan and execute actions, offering several benefits:

*   Works on unseen websites without custom code.
*   Resistant to website layout changes.
*   Applies workflows across numerous websites.
*   Uses LLMs for complex interactions.

For a detailed technical explanation, read the report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo
https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance and Evaluation

Skyvern boasts state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.  The detailed evaluation is available [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

### Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)
Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

## Advanced Usage

### Control your own browser (Chrome)

> ⚠️ WARNING: Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1.  **Just With Python Code**

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
2.  Make sure you don't have postgres running locally (Run `docker ps` to check)
3.  Clone the repository and navigate to the root directory
4.  Run `skyvern init llm` to generate a `.env` file. This will be copied into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml). *If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).*
6.  Run the following command via the commandline:
    ```bash
     docker compose up -d
    ```
7.  Navigate to `http://localhost:8080` in your browser to start using the UI

> **Important:** Only one Postgres container can run on port 5432 at a time. If you switch from the CLI-managed Postgres to Docker Compose, you must first remove the original container:
> ```bash
> docker rm -f postgresql-container
> ```

If you encounter any database related errors while using Docker to run Skyvern, check which Postgres container is running with `docker ps`.

## Skyvern Features

### Skyvern Tasks

Tasks are the core building blocks in Skyvern.  A task is a single instruction, specifying a `url`, `prompt`, and optional parameters like `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows

Workflows allow chaining multiple tasks to form a cohesive unit of work. Examples include downloading invoices, automating job applications, or purchasing items from an e-commerce site. Supported workflow features:

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

View the browser viewport in real-time for debugging and understanding.

### Form Filling

Native form-filling capabilities through `navigation_goal`.

### Data Extraction

Extract data from websites, with JSON schema support.

### File Downloading

Download files from websites, automatically uploaded to block storage.

### Authentication

Supports several authentication methods, including:

*   🔐 2FA Support (TOTP)
    Examples include:
    1.  QR-based 2FA (e.g. Google Authenticator, Authy)
    2.  Email based 2FA
    3.  SMS based 2FA
*   Password Manager Integrations
    - [x] Bitwarden
    - [ ] 1Password
    - [ ] LastPass

Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Model Context Protocol (MCP)

Supports the Model Context Protocol (MCP), enabling use with any LLM that supports it.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Connect Skyvern workflows to other apps via:

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-World Examples

Discover real-world applications of Skyvern:

### Invoice Downloading on many different websites
[Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif"/>
</p>

### Automate the job application process
[💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
<p align="center">
  <img src="fern/images/job_application_demo.gif"/>
</p>

### Automate materials procurement for a manufacturing company
[💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif"/>
</p>

### Navigating to government websites to register accounts or fill out forms
[💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
<p align="center">
  <img src="fern/images/edd_services.gif"/>
</p>
<!-- Add example of delaware entity lookups x2 -->

### Filling out random contact us forms
[💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
<p align="center">
  <img src="fern/images/contact_forms.gif"/>
</p>

### Retrieving insurance quotes from insurance providers in any language
[💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
<p align="center">
  <img src="fern/images/bci_seguros_recording.gif"/>
</p>

[💡 See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif"/>
</p>

## Contributor Setup

For a complete local environment CLI Installation

```bash
pip install -e .
```

The following command sets up your development environment to use pre-commit (our commit hook handler)

```
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` in your browser to start using the UI

    \*The Skyvern CLI supports Windows, WSL, macOS, and Linux environments.\*

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com).  For questions, contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                                                         |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                     |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                 |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                           |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                                                    |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)                             |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                                              |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable              | Description                              | Type      | Sample Value          |
| --------------------- | ---------------------------------------- | --------- | --------------------- |
| `ENABLE_OPENAI`       | Register OpenAI models                   | Boolean   | `true`, `false`       |
| `OPENAI_API_KEY`      | OpenAI API Key                           | String    | `sk-1234567890`       |
| `OPENAI_API_BASE`     | OpenAI API Base, optional                | String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional         | String    | `your-org-id`         |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable             | Description                     | Type      | Sample Value          |
| -------------------- | ------------------------------- | --------- | --------------------- |
| `ENABLE_ANTHROPIC`   | Register Anthropic models       | Boolean   | `true`, `false`       |
| `ANTHROPIC_API_KEY`  | Anthropic API key               | String    | `sk-1234567890`       |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable            | Description                           | Type      | Sample Value              |
| ------------------- | ------------------------------------- | --------- | ------------------------- |
| `ENABLE_AZURE`      | Register Azure OpenAI models          | Boolean   | `true`, `false`           |
| `AZURE_API_KEY`     | Azure deployment API key              | String    | `sk-1234567890`           |
| `AZURE_DEPLOYMENT`  | Azure OpenAI Deployment Name          | String    | `skyvern-deployment`      |
| `AZURE_API_BASE`    | Azure deployment api base url         | String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version                     | String    | `2024-02-01`              |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable             | Description                                                                                               | Type      | Sample Value          |
| -------------------- | --------------------------------------------------------------------------------------------------------- | --------- | --------------------- |
| `ENABLE_BEDROCK`     | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean   | `true`, `false`       |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable         | Description                | Type      | Sample Value               |
| ---------------- | -------------------------- | --------- | -------------------------- |
| `ENABLE_GEMINI`  | Register Gemini models     | Boolean   | `true`, `false`            |
| `GEMINI_API_KEY` | Gemini API Key             | String    | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable            | Description                        | Type      | Sample Value                          |
| ------------------- | ---------------------------------- | --------- | ------------------------------------- |
| `ENABLE_OLLAMA`     | Register local models via Ollama   | Boolean   | `true`, `false`                       |
| `OLLAMA_SERVER_URL` | URL for your Ollama server         | String    | `http://host.docker.internal:11434`   |
| `OLLAMA_MODEL`      | Ollama model name to load          | String    | `qwen2.5:7b-instruct`                 |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable             | Description                      | Type      | Sample Value           |
| -------------------- | -------------------------------- | --------- | ---------------------- |
| `ENABLE_OPENROUTER`  | Register OpenRouter models       | Boolean   | `true`, `false`        |
| `OPENROUTER_API_KEY` | OpenRouter API key               | String    | `sk-1234567890`        |
| `OPENROUTER_MODEL`   | OpenRouter model name            | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL          | String    | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                       | Description                               | Type      | Sample Value                    |
| ------------------------------ | ----------------------------------------- | --------- | ------------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`     | Register a custom OpenAI-compatible API endpoint | Boolean   | `true`, `false`                |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint | String    | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc. |
| `OPENAI_COMPATIBLE_API_KEY`    | API key for OpenAI-compatible endpoint    | String    | `sk-1234567890`                |
| `OPENAI_COMPATIBLE_API_BASE`   | Base URL for OpenAI-compatible endpoint   | String    | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc. |
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional | String    | `2023-05-15`                  |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional   | Integer   | `4096`, `8192`, etc.           |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional           | Float     | `0.0`, `0.5`, `0.7`, etc.       |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional | Boolean   | `true`, `false`                |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable                   | Description                       | Type      | Sample Value |
| -------------------------- | --------------------------------- | --------- | ------------ |
| `LLM_KEY`                  | The name of the model you want to use   | String    | See supported LLM keys above |
| `SECONDARY_LLM_KEY`        | The name of the model for mini agents skyvern runs with | String    | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS`    | Override the max tokens used by the LLM | Integer   | `128000`     |

## Feature Roadmap

*   \[x] Open Source
*   \[x] Workflow support
*   \[x] Improved context
*   \[x] Cost Savings
*   \[x] Self-serve UI
*   \[x] Workflow UI Builder
*   \[x] Chrome Viewport streaming
*   \[x] Past Runs UI
*   \[X] Auto workflow builder ("Observer") mode
*   \[x] Prompt Caching
*   \[x] Web Evaluation Dataset
*   \[ ] Improved Debug mode
*   \[ ] Chrome Extension
*   \[ ] Skyvern Action Recorder
*   \[ ] Interactable Livestream
*   \[ ] Integrate LLM Observability tools
*   \[x] Langchain Integration

## Contributing

Contributions are welcome!  Please see the [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for details. Contact us via email or discord.

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics by default. Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's open-source core is licensed under the [AGPL-3.0 License](LICENSE). Anti-bot measures within the managed cloud are not open-source.  Contact us with any licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)