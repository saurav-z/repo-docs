<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
</h1>

<p align="center">
  <strong>🤖 Automate Browser Workflows with LLMs and Computer Vision.</strong>
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

[Skyvern](https://github.com/Skyvern-AI/skyvern) is a powerful open-source tool that uses Large Language Models (LLMs) and computer vision to automate browser-based tasks, offering a robust alternative to traditional, fragile automation methods. Easily automate complex workflows on any website!

## Key Features

*   **No-Code Automation:** Automate tasks without writing custom code or relying on brittle DOM parsing.
*   **LLM-Powered:** Leverages LLMs to understand website layouts and reason through interactions.
*   **Website Agnostic:** Works on websites you've never seen before by mapping visual elements to actions.
*   **Robust to Changes:** Resistant to website layout changes; adapts to evolving web designs.
*   **Advanced Features:** Includes form filling, data extraction, file downloading, authentication, and more.
*   **Integrations:** Zapier, Make.com, and N8N integrations to connect to other apps.
*   **Livestreaming:** View a live stream of the browser actions.
*   **[Cloud & Local Options](#quickstart):** Easily run in the cloud via [Skyvern Cloud](https://app.skyvern.com) or locally.
*   **SOTA Performance:** Highest performance in the [WebBench benchmark](webbench.ai) with 64.4% accuracy, especially on WRITE tasks.

## Quickstart

### Skyvern Cloud

[Skyvern Cloud](https://app.skyvern.com) offers a managed cloud version of Skyvern. Run instances in parallel with features like anti-bot detection, proxy networks, and CAPTCHA solvers. Sign up at [app.skyvern.com](https://app.skyvern.com)

### Install & Run Locally

1.  **Install Skyvern:**

    ```bash
    pip install skyvern
    ```

2.  **Run Skyvern:**

    ```bash
    skyvern quickstart
    ```

3.  **Run a Task**
    *   **UI (Recommended):**

        ```bash
        skyvern run all
        ```
        Access the UI at http://localhost:8080 to run tasks.

    *   **Code:**

        ```python
        from skyvern import Skyvern

        skyvern = Skyvern()
        task = await skyvern.run_task(prompt="Find the top post on hackernews today")
        print(task)
        ```

        View task history at http://localhost:8080/history.

        Customize the API and base URL:

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

Skyvern is inspired by task-driven autonomous agent design from [BabyAGI](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT). It uses a swarm of agents and browser automation with [Playwright](https://playwright.dev/).

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
    <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram"/>
  </picture>

**Key Benefits:**

*   Works on unfamiliar websites.
*   Resistant to website layout changes.
*   Applies a single workflow across multiple sites.
*   LLMs enable complex reasoning.

Read the detailed technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<!-- Redo demo -->
[See Skyvern in action](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

## Performance & Evaluation

Skyvern leads in performance on the [WebBench benchmark](webbench.ai).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

Skyvern excels at WRITE tasks used for RPA (Robotic Process Automation).

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Task Performance"/>
</p>

## Advanced Usage

*   [Control Your Own Browser](#control-your-own-browser-chrome)
*   [Run with Any Remote Browser](#run-skyvern-with-any-remote-browser)
*   [Get Consistent Output Schema](#get-consistent-output-schema-from-your-run)
*   [Helpful Debugging Commands](#helpful-commands-to-debug-issues)
*   [Docker Compose Setup](#docker-compose-setup)

### Control Your Own Browser (Chrome)

> ⚠️ WARNING: [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port) requires copying your default user data.

1.  **With Python Code:**

    ```python
    from skyvern import Skyvern

    browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # Example Mac path
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

    *   Add these to your `.env` file:

        ```bash
        CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # Example Mac path
        BROWSER_TYPE=cdp-connect
        ```

    *   Restart the service: `skyvern run all`
    *   Run the task via UI or code.

### Run Skyvern with Any Remote Browser

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get Consistent Output Schema

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

### Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Check you have no local Postgres running with `docker ps`.
3.  Clone the repository and go to its root.
4.  Run `skyvern init llm` to generate a `.env` file.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml).
6.  Run `docker compose up -d`.
7.  Access the UI at `http://localhost:8080`.

> **Important:** Remove the Postgres container if using Docker Compose after running CLI-managed Postgres: `docker rm -f postgresql-container`.

If you encounter database errors, check the Postgres container with `docker ps`.

## Skyvern Features

### Skyvern Tasks

*   Each task is a specific instruction to navigate a website.
*   Tasks include `url`, `prompt`, and optional `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Screenshot"/>
</p>

### Skyvern Workflows

*   Chain tasks to build cohesive units of work.
*   Supports navigation, actions, data extraction, loops, and more.
*   Examples include invoice downloading, e-commerce automation, and more.

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Invoice Downloading Workflow Example"/>
</p>

### Livestreaming

*   View browser viewport live.

### Form Filling

*   Native form input capabilities via `navigation_goal`.

### Data Extraction

*   Extract data from websites and format the output using `data_extraction_schema`.

### File Downloading

*   Downloads and uploads files to block storage.

### Authentication

*   Supports authentication methods.

    *   🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).
    *   🔐 Password Manager Integrations (Bitwarden, more coming soon).

<p align="center">
  <img src="fern/images/secure_password_task_example.png" alt="Secure Password Task Example"/>
</p>

### 🔐 2FA Support (TOTP)
Skyvern supports a number of different 2FA methods to allow you to automate workflows that require 2FA.

Examples include:
1. QR-based 2FA (e.g. Google Authenticator, Authy)
1. Email based 2FA
1. SMS based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations
Skyvern currently supports the following password manager integrations:
- [x] Bitwarden
- [ ] 1Password
- [ ] LastPass

### Model Context Protocol (MCP)

Skyvern supports the Model Context Protocol (MCP) to allow you to use any LLM that supports MCP.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Skyvern supports Zapier, Make.com, and N8N to allow you to connect your Skyvern workflows to other apps.

* [Zapier](https://docs.skyvern.com/integrations/zapier)
* [Make.com](https://docs.skyvern.com/integrations/make.com)
* [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world Examples

*   **Invoice Downloading:** Automating invoice downloads across websites.

    <p align="center">
      <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Example"/>
    </p>

*   **Job Application Automation:** Apply for jobs automatically.
    [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

    <p align="center">
      <img src="fern/images/job_application_demo.gif" alt="Job Application Automation"/>
    </p>

*   **Materials Procurement:** Automate procurement processes.
    [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement Automation"/>
    </p>

*   **Government Website Automation:** Register accounts, fill forms.
    [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

    <p align="center">
      <img src="fern/images/edd_services.gif" alt="Government Website Automation"/>
    </p>

*   **Contact Form Filling:** Automatically fill out contact forms.
    [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

    <p align="center">
      <img src="fern/images/contact_forms.gif" alt="Contact Form Filling"/>
    </p>

*   **Insurance Quote Retrieval:** Retrieve quotes in any language.
    [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Retrieval"/>
    </p>

    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)

    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif" alt="Insurance Quote Retrieval"/>
    </p>

## Documentation

Comprehensive documentation is available on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

See list of supported models and configurations.

| Provider     | Supported Models                                                                                       |
| :----------- | :----------------------------------------------------------------------------------------------------- |
| OpenAI       | gpt4-turbo, gpt-4o, gpt-4o-mini                                                                        |
| Anthropic    | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                    |
| Azure OpenAI | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                |
| AWS Bedrock  | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                          |
| Gemini       | Gemini 2.5 Pro and flash, Gemini 2.0                                                                    |
| Ollama       | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)                               |
| OpenRouter   | Access models through [OpenRouter](https://openrouter.ai)                                             |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

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

We welcome contributions! See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For a high-level overview of Skyvern's structure and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics by default. Opt-out by setting `SKYVERN_TELEMETRY=false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE), excluding anti-bot measures in our cloud offering.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)