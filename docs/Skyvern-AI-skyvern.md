<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
</h1>

<p align="center">
  <strong>🚀 Automate complex browser-based workflows with the power of LLMs and Computer Vision, all while staying resistant to website changes. 🚀</strong>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Follow on Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="Follow on LinkedIn"/></a>
</p>

[Skyvern](https://www.skyvern.com) is your all-in-one solution for automating browser-based tasks using advanced LLMs and Computer Vision, providing a simple API to streamline any workflow.  [Check out the original repository](https://github.com/Skyvern-AI/skyvern)

## Key Features

*   **Intelligent Automation:** Automate workflows on websites you've never seen before, using Vision LLMs.
*   **Layout Change Resilience:** Built to withstand website updates by interacting visually, not just through code.
*   **Multi-Site Application:** Apply a single workflow across numerous websites by reasoning through interactions.
*   **Advanced Reasoning:** Leverage LLMs for complex scenarios, such as inferring answers and comparing products.
*   **Workflow Builder:** Create workflows by chaining multiple tasks for cohesive automation.
*   **Livestreaming:** Real-time browser view to monitor and debug tasks.
*   **Form Filling:** Native support for completing forms on websites.
*   **Data Extraction:** Extract specific data and output in structured formats.
*   **File Downloading:** Automated file download and storage.
*   **Authentication Support:** Robust support for a variety of authentication methods, including 2FA.
*   **Model Context Protocol (MCP) Support:** Integrate with any LLM that supports MCP.
*   **Integration with Zapier, Make.com, and N8N**: Easy to connect with other applications.

## Quickstart

### Skyvern Cloud
[Skyvern Cloud](https://app.skyvern.com) provides a managed cloud version of Skyvern, complete with anti-bot and CAPTCHA solutions.

### Install & Run

**Prerequisites:**
*   [Python 3.11.x](https://www.python.org/downloads/), works with 3.12, not ready yet for 3.13
*   [NodeJS & NPM](https://nodejs.org/en/download/)

**Windows-Specific:**
*   [Rust](https://rustup.rs/)
*   VS Code with C++ dev tools and Windows SDK

1.  **Install Skyvern:**
    ```bash
    pip install skyvern
    ```

2.  **Quickstart (Initial Setup):**
    ```bash
    skyvern quickstart
    ```

3.  **Run Tasks (UI Recommended):**
    ```bash
    skyvern run all
    ```
    Then, navigate to [http://localhost:8080](http://localhost:8080) to use the UI.

    **Code Example:**
    ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
    ```

    Or run on Cloud or Service:
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

*   **Control Your Own Browser (Chrome):**  Specify the browser path in your code or environment variables.
*   **Run Skyvern with Any Remote Browser:**  Use a CDP connection URL.
*   **Consistent Output Schema:**  Use the `data_extraction_schema` parameter.
*   **Debugging Commands:**  `skyvern run server`, `skyvern run ui`, `skyvern status`, `skyvern stop all`, `skyvern stop ui`, `skyvern stop server`

## Docker Compose Setup

1.  Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2.  No local Postgres instance
3.  Clone the repo
4.  Run `skyvern init llm`
5.  Fill LLM API key in `docker-compose.yml`
6.  `docker compose up -d`
7.  Visit `http://localhost:8080`

> **Important:** Remove old container with `docker rm -f postgresql-container` if switching Postgres container.

## Skyvern Features (In-Depth)

### Skyvern Tasks

Tasks are the fundamental unit, defining a website, prompt, and optional data schema and error codes.

### Skyvern Workflows

Workflows chain tasks together. Features include:

*   Browser Task
*   Browser Action
*   Data Extraction
*   Validation
*   For Loops
*   File parsing
*   Sending emails
*   Text Prompts
*   HTTP Request Block
*   Custom Code Block
*   Uploading files to block storage
*   (Coming soon) Conditionals

<p align="center">
  <img src="fern/images/block_example_v2.png"/>
</p>

### Livestreaming

View the browser actions in real-time.

### Form Filling

Native support to fill form inputs.

### Data Extraction

Extract specific data. Use a `data_extraction_schema` for structured output.

### File Downloading

Files automatically saved to block storage.

### Authentication

Supports various authentication methods and password manager integrations.

#### 🔐 2FA Support (TOTP)

*   QR-based 2FA (e.g. Google Authenticator, Authy)
*   Email based 2FA
*   SMS based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Integrates with any LLM that supports MCP.

### Zapier / Make.com / N8N Integration

Supports Zapier, Make.com, and N8N to connect Skyvern workflows.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world Examples

*   [Invoice Downloading](https://meetings.hubspot.com/skyvern/demo)
    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   [Job Application](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>
*   [Materials Procurement](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>
*   [Government Website Automation](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>
*   [Contact Us Forms](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>
*   [Insurance Quotes](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Documentation

Comprehensive documentation is available on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

| Provider        | Supported Models                                   |
| --------------- | -------------------------------------------------- |
| OpenAI          | gpt4-turbo, gpt-4o, gpt-4o-mini                   |
| Anthropic       | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Azure OpenAI    | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                   |
| AWS Bedrock     | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)      |
| Gemini          | Gemini 2.5 Pro and flash, Gemini 2.0                   |
| Ollama          | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter      | Access models through [OpenRouter](https://openrouter.ai) |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

**OpenAI**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OPENAI`| Register OpenAI models | Boolean | `true`, `false` |
| `OPENAI_API_KEY` | OpenAI API Key | String | `sk-1234567890` |
| `OPENAI_API_BASE` | OpenAI API Base, optional | String | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional | String | `your-org-id` |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

**Anthropic**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_ANTHROPIC` | Register Anthropic models| Boolean | `true`, `false` |
| `ANTHROPIC_API_KEY` | Anthropic API key| String | `sk-1234567890` |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

**Azure OpenAI**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_AZURE` | Register Azure OpenAI models | Boolean | `true`, `false` |
| `AZURE_API_KEY` | Azure deployment API key | String | `sk-1234567890` |
| `AZURE_DEPLOYMENT` | Azure deployment API key | String | `skyvern-deployment`|
| `AZURE_API_BASE` | Azure deployment api base url| String | `https://skyvern-deployment.openai.azure.com/`|
| `AZURE_API_VERSION` | Azure API Version| String | `2024-02-01`|

Recommended `LLM_KEY`: `AZURE_OPENAI`

**AWS Bedrock**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_BEDROCK` | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false` |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

**Gemini**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_GEMINI` | Register Gemini models| Boolean | `true`, `false` |
| `GEMINI_API_KEY` | Gemini API Key| String | `your_google_gemini_api_key`|

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

**Ollama**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OLLAMA`| Register local models via Ollama | Boolean | `true`, `false` |
| `OLLAMA_SERVER_URL` | URL for your Ollama server | String | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Ollama model name to load | String | `qwen2.5:7b-instruct` |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

**OpenRouter**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `ENABLE_OPENROUTER`| Register OpenRouter models | Boolean | `true`, `false` |
| `OPENROUTER_API_KEY` | OpenRouter API key | String | `sk-1234567890` |
| `OPENROUTER_MODEL` | OpenRouter model name | String | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | String | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

**OpenAI-Compatible**
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

**General LLM Configuration**
| Variable | Description| Type | Sample Value|
| -------- | ------- | ------- | ------- |
| `LLM_KEY` | The name of the model you want to use | String | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | The name of the model for mini agents skyvern runs with | String | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |

## Feature Roadmap

*   [x] Open Source & Key Features
*   [ ] Improved Debug mode
*   [ ] Chrome Extension
*   [ ] Skyvern Action Recorder
*   [ ] Interactable Livestream
*   [ ] Integrate LLM Observability tools
*   [x] Langchain Integration

## Contributing

See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

Use [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme) for guidance.

## Telemetry

Opt-out by setting `SKYVERN_TELEMETRY=false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)