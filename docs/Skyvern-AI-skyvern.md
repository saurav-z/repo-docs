<!-- DOCTOC SKIP -->

<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
    </picture>
  </a>
  <br />
  <br />
  <p><b>Automate browser workflows effortlessly with Skyvern, your AI-powered browser automation solution.</b></p>
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

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern in Action"/>
</p>

## Key Features

*   **AI-Powered Automation:** Leverage Large Language Models (LLMs) and computer vision to automate complex browser-based tasks.
*   **Robustness:** Avoid brittle, code-dependent automation by relying on vision-based interactions.
*   **Versatility:** Apply a single workflow across numerous websites without custom scripting.
*   **Workflow Capabilities:** Chain tasks, extract data, loop, and more to create sophisticated automations.
*   **Real-time Monitoring:**  Livestream browser activity for debugging and oversight.
*   **Form Filling & Data Extraction:** Seamlessly fill forms and extract structured data from websites.
*   **Authentication Support:** Automate tasks behind logins with various authentication methods, including 2FA.
*   **Integrations:** Integrate Skyvern with Zapier, Make.com, and N8N for expanded functionality.

For more information and examples, visit the [Skyvern GitHub repository](https://github.com/Skyvern-AI/skyvern).

## Quickstart

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern (Choose one)

#### a. Skyvern Cloud (Managed Service)

[Skyvern Cloud](https://app.skyvern.com) offers a managed cloud version for immediate use, eliminating infrastructure setup.

#### b. Local Setup

```bash
skyvern quickstart
```

### 3. Run a Task

#### a. UI (Recommended)

Start the Skyvern service and access the UI.

```bash
skyvern run all
```

Open http://localhost:8080 to run a task via the UI.

#### b. Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

The browser opens, executes the task, and closes automatically. View task history at http://localhost:8080/history.

You can also specify the target where you would like to run Skyvern.
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

Inspired by task-driven autonomous agent design, Skyvern uses LLMs and computer vision combined with browser automation libraries like [Playwright](https://playwright.dev/) to navigate the web. Skyvern's architecture consists of:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" />
</picture>

Key advantages:

*   **Website Agnostic:** Works on new and unseen websites.
*   **Layout Resilience:** Adaptable to website layout changes.
*   **Scalability:**  Applies the same workflow to multiple websites.
*   **Intelligent Reasoning:** Handles complex scenarios, such as inferring information.

Read more in our detailed technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Performance & Evaluation

Skyvern is on the cutting edge of the browser automation space, and is tested on the WebBench benchmark.

### [WebBench Benchmark](webbench.ai)
Skyvern demonstrates SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="Webbench Overall Performance"/>
</p>

### [WebBench WRITE tasks](webbench.ai)
Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="Webbench Write Tasks Performance"/>
</p>

## Advanced Usage

### 1. Control your own browser (Chrome)
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

### 2. Run Skyvern with any remote browser
Grab the cdp connection url and pass it to Skyvern

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### 3. Get consistent output schema from your run
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

### 4. Helpful commands to debug issues
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
2.  Confirm no local PostgreSQL instance is running (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file. This is copied into the Docker image.
5.  Populate the LLM provider key in [docker-compose.yml](./docker-compose.yml).  Set the correct server IP for the UI container if deploying to a remote server.
6.  Execute:

    ```bash
    docker compose up -d
    ```

7.  Access the UI at `http://localhost:8080`.

> **Important:** Only one Postgres container can run on port 5432 at once.  If you switch from the CLI-managed Postgres to Docker Compose, first remove the original container:
> ```bash
> docker rm -f postgresql-container
> ```

If you encounter database errors, check running containers with `docker ps`.

## Skyvern Features in Detail

### Skyvern Tasks
Tasks are the core building blocks within Skyvern, representing individual requests to navigate and accomplish goals on websites.

Tasks require you to specify a `url`, `prompt`, and optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Screenshot"/>
</p>

### Skyvern Workflows
Workflows allow you to chain multiple tasks together to create a cohesive process.

Examples: Download all invoices, automate product purchasing, or automate the job application process.

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
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Invoice Downloading Workflow Example"/>
</p>

### Livestreaming
View the browser viewport live on your local machine for debugging and understanding Skyvern's interactions.

### Form Filling
Skyvern can natively fill out form inputs on websites, by passing `navigation_goal`.

### Data Extraction
Skyvern is capable of extracting data from a website.

You can also specify a `data_extraction_schema` directly within the main prompt to tell Skyvern exactly what data you'd like to extract from the website, in jsonc format. Skyvern's output will be structured in accordance to the supplied schema.

### File Downloading
Skyvern can download files, automatically uploading them to block storage if configured, accessible via the UI.

### Authentication
Skyvern supports multiple authentication methods.

### 🔐 2FA Support (TOTP)
Skyvern supports 2FA methods:

1.  QR-based 2FA (e.g. Google Authenticator, Authy)
2.  Email-based 2FA
3.  SMS-based 2FA

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

## Real-world Examples

See Skyvern in action:

### Invoice Downloading

[Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading on many different websites"/>
</p>

### Automate the job application process

[💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Automate the job application process"/>
</p>

### Automate materials procurement for a manufacturing company

[💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Automate materials procurement for a manufacturing company"/>
</p>

### Navigating to government websites to register accounts or fill out forms

[💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="Navigating to government websites to register accounts or fill out forms"/>
</p>

### Filling out random contact us forms

[💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Filling out random contact us forms"/>
</p>

### Retrieving insurance quotes from insurance providers in any language

[💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="Retrieving insurance quotes from insurance providers in any language"/>
</p>

[💡 See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Retrieving insurance quotes from insurance providers in any language"/>
</p>

## Contributor Setup

Install with:

```bash
pip install -e .
```

Setup your development environment:

```bash
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` to use the UI.
    *Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Find detailed documentation on our [📕 docs page](https://docs.skyvern.com).  For support, contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                                |
| ------------- | --------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)            |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)  |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                           |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)       |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable            | Description                                  | Type      | Sample Value           |
| ------------------- | -------------------------------------------- | --------- | ---------------------- |
| `ENABLE_OPENAI`     | Register OpenAI models                     | Boolean   | `true`, `false`        |
| `OPENAI_API_KEY`    | OpenAI API Key                             | String    | `sk-1234567890`        |
| `OPENAI_API_BASE`   | OpenAI API Base, optional                  | String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional            | String    | `your-org-id`          |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable            | Description                        | Type      | Sample Value           |
| ------------------- | ---------------------------------- | --------- | ---------------------- |
| `ENABLE_ANTHROPIC`  | Register Anthropic models          | Boolean   | `true`, `false`        |
| `ANTHROPIC_API_KEY` | Anthropic API key                  | String    | `sk-1234567890`        |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable             | Description                                | Type      | Sample Value                           |
| -------------------- | ------------------------------------------ | --------- | -------------------------------------- |
| `ENABLE_AZURE`       | Register Azure OpenAI models               | Boolean   | `true`, `false`                        |
| `AZURE_API_KEY`      | Azure deployment API key                   | String    | `sk-1234567890`                        |
| `AZURE_DEPLOYMENT`   | Azure OpenAI Deployment Name               | String    | `skyvern-deployment`                   |
| `AZURE_API_BASE`     | Azure deployment api base url              | String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`  | Azure API Version                        | String    | `2024-02-01`                           |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable             | Description                                                                                                                                                             | Type      | Sample Value           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------------------- |
| `ENABLE_BEDROCK`     | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean   | `true`, `false`        |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable            | Description                    | Type      | Sample Value            |
| ------------------- | ------------------------------ | --------- | ----------------------- |
| `ENABLE_GEMINI`     | Register Gemini models         | Boolean   | `true`, `false`         |
| `GEMINI_API_KEY`    | Gemini API Key                 | String    | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable              | Description                      | Type      | Sample Value               |
| --------------------- | -------------------------------- | --------- | -------------------------- |
| `ENABLE_OLLAMA`       | Register local models via Ollama | Boolean   | `true`, `false`            |
| `OLLAMA_SERVER_URL`   | URL for your Ollama server       | String    | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`        | Ollama model name to load        | String    | `qwen2.5:7b-instruct`      |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable              | Description                      | Type      | Sample Value               |
| --------------------- | -------------------------------- | --------- | -------------------------- |
| `ENABLE_OPENROUTER`   | Register OpenRouter models       | Boolean   | `true`, `false`            |
| `OPENROUTER_API_KEY`  | OpenRouter API key               | String    | `sk-1234567890`            |
| `OPENROUTER_MODEL`    | OpenRouter model name            | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL          | String    | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                         | Description                                 | Type      | Sample Value                                   |
| -------------------------------- | ------------------------------------------- | --------- | ---------------------------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`       | Register a custom OpenAI-compatible API endpoint | Boolean   | `true`, `false`                                |
| `OPENAI_COMPATIBLE_MODEL_NAME`   | Model name for OpenAI-compatible endpoint  | String    | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc. |
| `OPENAI_COMPATIBLE_API_KEY`      | API key for OpenAI-compatible endpoint     | String    | `sk-1234567890`                                |
| `OPENAI_COMPATIBLE_API_BASE`     | Base URL for OpenAI-compatible endpoint    | String    | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc. |
| `OPENAI_COMPATIBLE_API_VERSION`  | API version for OpenAI-compatible endpoint, optional| String | `2023-05-15`                                  |
| `OPENAI_COMPATIBLE_MAX_TOKENS`   | Maximum tokens for completion, optional    | Integer   | `4096`, `8192`, etc.                           |
| `OPENAI_COMPATIBLE_TEMPERATURE`  | Temperature setting, optional             | Float     | `0.0`, `0.5`, `0.7`, etc.                      |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional   | Boolean   | `true`, `false`                                |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable              | Description                             | Type      | Sample Value           |
| --------------------- | --------------------------------------- | --------- | ---------------------- |
| `LLM_KEY`             | The name of the model you want to use    | String    | See supported LLM keys above |
| `SECONDARY_LLM_KEY`   | The name of the model for mini agents  | String    | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer   | `128000`               |

## Feature Roadmap

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

We welcome contributions! Open a PR/issue or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).  Refer to our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For a high-level overview, explore [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

By default, Skyvern collects basic usage data. To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's core code is licensed under the [AGPL-3.0 License](LICENSE). The anti-bot measures are excluded, existing only within our managed cloud.  Contact us at [support@skyvern.com](mailto:support@skyvern.com) with any licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)