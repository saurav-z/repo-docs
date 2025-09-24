<div align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
    </picture>
  </a>
  <h1>Skyvern: Automate Your Browser Workflows with AI</h1>
  <p><b>Effortlessly automate complex browser tasks using Large Language Models (LLMs) and computer vision.</b></p>
  <p>
    <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black&style=flat-square"/></a>
    <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black&style=flat-square"/></a>
    <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord&style=flat-square"/></a>
    <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern?style=flat-square" alt="GitHub Stars"/></a>
    <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern?style=flat-square"/></a>
    <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social&style=flat-square"/></a>
    <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin&style=flat-square"/></a>
  </p>
</div>

[Skyvern](https://www.skyvern.com) empowers you to automate browser-based tasks by leveraging the power of LLMs and computer vision.  Say goodbye to brittle, website-specific automation scripts and embrace a more robust and adaptable solution. <b><a href="https://github.com/Skyvern-AI/skyvern">Check out the original repo</a></b>

## Key Features

*   🤖 **AI-Powered Automation:** Use LLMs and computer vision to interact with websites as if they were a human user.
*   🌐 **Cross-Website Compatibility:** Works on websites you've never seen before without custom code.
*   🔄 **Robustness:** Resilient to website layout changes.
*   💡 **Intelligent Reasoning:**  Leverages LLMs to handle complex situations, like inferring answers to questions.
*   ✅ **Workflow Automation:** Create and run automated workflows by chaining together tasks with support for loops, conditional logic, file parsing, and more.
*   🎥 **Live Viewport Streaming:**  See exactly how Skyvern interacts with websites.
*   📄 **Data Extraction & Form Filling:** Effortlessly extract data and fill out forms.
*   🔐 **Authentication Support:** Automate tasks behind logins, including 2FA (TOTP, QR-based, email, SMS) with password manager integrations.
*   ☁️ **Skyvern Cloud:**  Fully-managed cloud version for easy deployment.
*   ⚙️ **Integrations:** Zapier, Make.com, and N8N integration.

## How Skyvern Works

Skyvern uses a task-driven, agent-based design inspired by BabyAGI and AutoGPT. It employs a swarm of agents to:

1.  Comprehend the website.
2.  Plan actions.
3.  Execute those actions using browser automation libraries like Playwright.

### System Diagram
![Skyvern 2.0 System Diagram](fern/images/skyvern_2_0_system_diagram.png)

### Advantages:

*   Operates on unseen websites.
*   Adapts to website layout changes.
*   Applies a single workflow across many sites.
*   Utilizes LLMs for intelligent reasoning.

## Performance & Evaluation

Skyvern achieves state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai), and excels at WRITE tasks (e.g., filling forms, logins), demonstrating significant improvement in Robotic Process Automation (RPA) scenarios.

### WebBench Performance

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

### WRITE Task Performance

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Task Performance"/>
</p>

Read the detailed technical report + evaluation [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

## Quickstart

### Skyvern Cloud
Try the fully-managed cloud version at [app.skyvern.com](https://app.skyvern.com).

### Install & Run Locally

Dependencies:

*   [Python 3.11.x](https://www.python.org/downloads/) (works with 3.12, not ready for 3.13)
*   [NodeJS & NPM](https://nodejs.org/en/download/)

Additional (for Windows):
*   [Rust](https://rustup.rs/)
*   VS Code with C++ dev tools and Windows SDK

1.  **Install:**

    ```bash
    pip install skyvern
    ```
2.  **Quickstart:**

    This initializes the database and runs migrations.

    ```bash
    skyvern quickstart
    ```
3.  **Run Task:**

    *   **UI (Recommended):**
        Start the service and UI (after DB setup).

        ```bash
        skyvern run all
        ```

        Access the UI at `http://localhost:8080` to run tasks.

    *   **Code:**

        ```python
        from skyvern import Skyvern

        skyvern = Skyvern()
        task = await skyvern.run_task(prompt="Find the top post on hackernews today")
        print(task)
        ```

        View task history at `http://localhost:8080/history`.

        You can also specify the run target:

        ```python
        from skyvern import Skyvern

        # Cloud
        skyvern = Skyvern(api_key="SKYVERN API KEY")

        # Local Service
        skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

        task = await skyvern.run_task(prompt="Find the top post on hackernews today")
        print(task)
        ```
## Advanced Usage

### 💻 Control Your Browser (Chrome)

> ⚠️ Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser.

1.  **With Python Code:**

    ```python
    from skyvern import Skyvern

    browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" # Mac
    skyvern = Skyvern(
        base_url="http://localhost:8000",
        api_key="YOUR_API_KEY",
        browser_path=browser_path,
    )
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    ```

2.  **With Skyvern Service:**

    Add to your `.env` file:

    ```bash
    CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" # Mac
    BROWSER_TYPE=cdp-connect
    ```

    Restart the service: `skyvern run all`. Run the task via UI or code.

### 🌐 Run With a Remote Browser

Get the CDP connection URL and pass it:

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
```

### 📝 Get Consistent Output Schema

Use `data_extraction_schema`:

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
    data_extraction_schema={...}  # Your JSON schema
)
```
### ⚙️ Debugging Commands

```bash
# Start Skyvern Server Separately
skyvern run server

# Start Skyvern UI
skyvern run ui

# Check Status
skyvern status

# Stop Services
skyvern stop all
skyvern stop ui
skyvern stop server
```

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  No local Postgres is running. Use `docker ps` to check.
3.  Clone the repo.
4.  Run `skyvern init llm` to create a `.env` file and specify your LLM key.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml).
6.  Run:

    ```bash
    docker compose up -d
    ```
7.  Open `http://localhost:8080` in your browser.

> **Important:** If switching from CLI-managed Postgres to Docker, remove the original container first: `docker rm -f postgresql-container`.

## Skyvern Features in Detail

### Skyvern Tasks

The core building blocks. Each task is a single request to Skyvern to achieve a specific goal on a website.

Tasks require a `url`, `prompt`, and optionally a `data schema` and `error codes`.

### Skyvern Workflows

Chain multiple tasks together to automate complex processes. Features include:

1.  Browser Tasks
2.  Browser Actions
3.  Data Extraction
4.  Validation
5.  For Loops
6.  File Parsing
7.  Sending Emails
8.  Text Prompts
9.  HTTP Request Block
10. Custom Code Block
11. Uploading files to block storage
12.  (Coming soon) Conditionals

### Livestreaming

Real-time viewing of the browser viewport for debugging and oversight.

### Form Filling

Native form input support using information provided in the `navigation_goal`.

### Data Extraction

Extract data from a website.
*   Specify a `data_extraction_schema` to structure the output.

### File Downloading

Download files, which are automatically uploaded to block storage (if configured) and accessible via the UI.

### Authentication

Supports various authentication methods:

*   2FA Support (TOTP, QR-based, email, SMS): [Learn more](https://docs.skyvern.com/credentials/totp).
*   Password Manager Integrations: Bitwarden support (1Password and LastPass coming).

### Model Context Protocol (MCP)

Integrate with any LLM supporting the MCP: [See MCP documentation](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Connect Skyvern workflows to other apps:

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

## Real-World Examples

*   Invoice Downloading ([Book a demo](https://meetings.hubspot.com/skyvern/demo))
    ![Invoice Downloading Demo](fern/images/invoice_downloading.gif)
*   Automating Job Applications
    [See in action](https://app.skyvern.com/tasks/create/job_application)
    ![Job Application Demo](fern/images/job_application_demo.gif)
*   Procurement for Manufacturing
    [See in action](https://app.skyvern.com/tasks/create/finditparts)
    ![FinditParts Recording](fern/images/finditparts_recording_crop.gif)
*   Government Website Automation
    [See in action](https://app.skyvern.com/tasks/create/california_edd)
    ![EDD Services](fern/images/edd_services.gif)
*   Contact Us Form Filling
    [See in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    ![Contact Forms](fern/images/contact_forms.gif)
*   Insurance Quote Retrieval
    [See in action](https://app.skyvern.com/tasks/create/bci_seguros)
    ![BCI Seguros Recording](fern/images/bci_seguros_recording.gif)
    [See in action](https://app.skyvern.com/tasks/create/geico)
    ![Geico Shu Recording](fern/images/geico_shu_recording_cropped.gif)


## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

| Provider      | Supported Models                                                               |
| ------------- | ------------------------------------------------------------------------------ |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                               |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                         |
| Azure OpenAI  | Any GPT models (Best with multimodal LLMs, e.g. azure/gpt4-o)                   |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                  |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                          |
| Ollama        | Locally hosted models via [Ollama](https://github.com/ollama/ollama)           |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                     |
| OpenAI-compatible | Any API endpoint following OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

### LLM Configuration (Environment Variables)

See the original README for detailed environment variable settings.

##### OpenAI
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OPENAI`| Register OpenAI models | Boolean | `true`, `false` |
| `OPENAI_API_KEY` | OpenAI API Key | String | `sk-1234567890` |
| `OPENAI_API_BASE` | OpenAI API Base, optional | String | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional | String | `your-org-id` |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_ANTHROPIC` | Register Anthropic models| Boolean | `true`, `false` |
| `ANTHROPIC_API_KEY` | Anthropic API key| String | `sk-1234567890` |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_AZURE` | Register Azure OpenAI models | Boolean | `true`, `false` |
| `AZURE_API_KEY` | Azure deployment API key | String | `sk-1234567890` |
| `AZURE_DEPLOYMENT` | Azure OpenAI Deployment Name | String | `skyvern-deployment`|
| `AZURE_API_BASE` | Azure deployment api base url| String | `https://skyvern-deployment.openai.azure.com/`|
| `AZURE_API_VERSION` | Azure API Version| String | `2024-02-01`|

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_BEDROCK` | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false` |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_GEMINI` | Register Gemini models| Boolean | `true`, `false` |
| `GEMINI_API_KEY` | Gemini API Key| String | `your_google_gemini_api_key`|

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OLLAMA`| Register local models via Ollama | Boolean | `true`, `false` |
| `OLLAMA_SERVER_URL` | URL for your Ollama server | String | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Ollama model name to load | String | `qwen2.5:7b-instruct` |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OPENROUTER`| Register OpenRouter models | Boolean | `true`, `false` |
| `OPENROUTER_API_KEY` | OpenRouter API key | String | `sk-1234567890` |
| `OPENROUTER_MODEL` | OpenRouter model name | String | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | String | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OPENAI_COMPATIBLE`| Register a custom OpenAI-compatible API endpoint | Boolean | `true`, `false` |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint | String | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY` | API key for OpenAI-compatible endpoint | String | `sk-1234567890`|
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint | String | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional| String | `2023-05-15`|
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional| Integer | `4096`, `8192`, etc.|
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional| Float | `0.0`, `0.5`, `0.7`, etc.|
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional| Boolean | `true`, `false`|

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `LLM_KEY` | The name of the model you want to use | String | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | The name of the model for mini agents skyvern runs with | String | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |
## Feature Roadmap

*   [x] Open Source
*   [x] Workflow support
*   [x] Improved context
*   [x] Cost Savings
*   [x] Self-serve UI
*   [x] Workflow UI Builder
*   [x] Chrome Viewport streaming
*   [x] Past Runs UI
*   [x] Auto workflow builder ("Observer" mode)
*   [x] Prompt Caching
*   [x] Web Evaluation Dataset
*   [ ] Improved Debug mode
*   [ ] Chrome Extension
*   [ ] Skyvern Action Recorder
*   [ ] Interactable Livestream
*   [ ] Integrate LLM Observability tools
*   [x] Langchain Integration

## Contributing

We welcome contributions! See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started. Contact the [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme) for project help.

## Telemetry

Skyvern collects basic usage statistics. To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's core logic is licensed under the [AGPL-3.0 License](LICENSE). Managed cloud offering has additional anti-bot measures.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)