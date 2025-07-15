<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
</h1>

<p align="center">
  <b>Automate any browser-based workflow with the power of LLMs and Computer Vision.</b>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

[Skyvern](https://www.skyvern.com) is a powerful, open-source tool that automates browser-based workflows using Large Language Models (LLMs) and computer vision, allowing you to automate repetitive tasks and build sophisticated automations.

**Key Features:**

*   **Intuitive Automation:** Automate workflows without custom scripting or reliance on brittle DOM parsing.
*   **LLM-Driven Interactions:** Leverage Vision LLMs to understand and interact with websites, adapting to layout changes.
*   **Workflow Builder:** Chain multiple tasks together to create complex, automated workflows.
*   **Data Extraction:** Extract structured data from websites with data extraction schemas.
*   **Advanced Features:** Form filling, file downloads, authentication, and more.
*   **Open Source:** Benefit from a fully open-source solution with a vibrant community.

**Ready to get started?**  <br>
Check out the original repository: [https://github.com/Skyvern-AI/skyvern](https://github.com/Skyvern-AI/skyvern)

## Quickstart

### 1. Installation

```bash
pip install skyvern
```

### 2. Run Skyvern (UI)

```bash
skyvern run all
```

Open your browser and go to `http://localhost:8080` to run tasks using the UI.

### 3. Run Skyvern (Code)

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## How Skyvern Works

Skyvern employs a task-driven autonomous agent design inspired by BabyAGI and AutoGPT, but with browser automation using libraries such as Playwright. This allows the system to interact with websites it has never seen before, mapping visual elements to actions, and resisting website layout changes.

**Key Advantages:**

*   **Adaptability:** Operates on unseen websites, adapting to layout changes.
*   **Scalability:** Applies workflows across numerous websites.
*   **Reasoning:** Utilizes LLMs for complex interaction and inference (e.g., driver's license age).

For detailed technical information, see the report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Performance & Evaluation

Skyvern achieves state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

### WRITE Task Performance

Skyvern excels in WRITE tasks (RPA-related) such as form filling and downloads.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Task Performance"/>
</p>

## Advanced Usage

*   **Control Your Browser:** Utilize your local Chrome browser for tasks.
*   **Remote Browser Integration:** Connect to remote browsers via CDP.
*   **Structured Output:** Specify `data_extraction_schema` for consistent data formats.
*   **Debugging Commands:**
    *   `skyvern run server`
    *   `skyvern run ui`
    *   `skyvern status`
    *   `skyvern stop all`
    *   `skyvern stop ui`
    *   `skyvern stop server`

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Ensure no local Postgres instance is running (check with `docker ps`).
3.  Clone the repository.
4.  Run `skyvern init llm` to create a `.env` file.
5.  Fill in the LLM provider keys in [docker-compose.yml](./docker-compose.yml). Set the correct server IP in this file.
6.  Run `docker compose up -d`.
7.  Access the UI at `http://localhost:8080`.

> **Important:** Remove any existing Postgres container before running Docker Compose: `docker rm -f postgresql-container`.

## Skyvern Features - Deep Dive

### Skyvern Tasks

*   The fundamental building blocks.
*   Requires `url`, `prompt`, and optionally `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Screenshot"/>
</p>

### Skyvern Workflows

*   Chain multiple tasks.
*   Features include:
    *   Navigation
    *   Action
    *   Data Extraction
    *   Loops
    *   File parsing
    *   File Uploads to block storage
    *   Sending emails
    *   Text Prompts
    *   Tasks (general)
    *   (Coming soon) Conditionals
    *   (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Invoice Downloading Workflow Example"/>
</p>

### Livestreaming

*   View the browser viewport in real time for debugging and intervention.

### Form Filling

*   Native form input capabilities.

### Data Extraction

*   Extract structured data.
*   Define `data_extraction_schema` for JSON output.

### File Downloading

*   Download files automatically.
*   Files are uploaded to block storage (if configured).

### Authentication

*   Supports various authentication methods.

#### 🔐 2FA Support (TOTP)

*   QR-based 2FA (Google Authenticator, Authy)
*   Email-based 2FA
*   SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

*   Supports MCP for using any LLM that supports it.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

*   Integrate Skyvern with other apps.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world examples of Skyvern

*   **Invoice Downloading:** [Book a demo](https://meetings.hubspot.com/skyvern/demo)
<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo"/>
</p>
*   **Job Application Automation:** [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Automation Demo"/>
</p>
*   **Materials Procurement Automation:** [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement Automation Demo"/>
</p>
*   **Government Website Automation:** [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
<p align="center">
  <img src="fern/images/edd_services.gif" alt="Government Website Automation Demo"/>
</p>
*   **Contact Us Form Filling:** [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Us Form Filling Demo"/>
</p>
*   **Insurance Quote Retrieval:** [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Retrieval Demo"/>
</p>

[💡 See it in action](https://app.skyvern.com/tasks/create/geico)
<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Quote Demo"/>
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

    *The Skyvern CLI supports Windows, WSL, macOS, and Linux environments.*

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com). For any clarification or missing information, please open an issue or reach out [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                  |
| ------------- | -------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                    |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                 |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai) |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable                 | Description                    | Type    | Sample Value        |
| ------------------------ | ------------------------------ | ------- | ------------------- |
| `ENABLE_OPENAI`          | Register OpenAI models         | Boolean | `true`, `false`     |
| `OPENAI_API_KEY`         | OpenAI API Key                 | String  | `sk-1234567890`     |
| `OPENAI_API_BASE`        | OpenAI API Base, optional      | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION`   | OpenAI Organization ID, optional | String  | `your-org-id`       |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable            | Description              | Type    | Sample Value        |
| ------------------- | ------------------------ | ------- | ------------------- |
| `ENABLE_ANTHROPIC`  | Register Anthropic models | Boolean | `true`, `false`     |
| `ANTHROPIC_API_KEY` | Anthropic API key        | String  | `sk-1234567890`     |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable           | Description                       | Type    | Sample Value                  |
| ------------------ | --------------------------------- | ------- | ----------------------------- |
| `ENABLE_AZURE`     | Register Azure OpenAI models      | Boolean | `true`, `false`               |
| `AZURE_API_KEY`    | Azure deployment API key          | String  | `sk-1234567890`               |
| `AZURE_DEPLOYMENT` | Azure OpenAI Deployment Name    | String  | `skyvern-deployment`          |
| `AZURE_API_BASE`   | Azure deployment api base url   | String  | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`| Azure API Version               | String  | `2024-02-01`                  |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable                | Description                                                                                                                            | Type    | Sample Value        |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------- |
| `ENABLE_BEDROCK`        | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean | `true`, `false`     |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable           | Description             | Type    | Sample Value            |
| ------------------ | ----------------------- | ------- | ----------------------- |
| `ENABLE_GEMINI`    | Register Gemini models   | Boolean | `true`, `false`         |
| `GEMINI_API_KEY`   | Gemini API Key          | String  | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable             | Description              | Type    | Sample Value           |
| -------------------- | ------------------------ | ------- | ---------------------- |
| `ENABLE_OLLAMA`      | Register local models via Ollama | Boolean | `true`, `false`        |
| `OLLAMA_SERVER_URL`  | URL for your Ollama server | String  | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`       | Ollama model name to load| String  | `qwen2.5:7b-instruct`  |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable             | Description              | Type    | Sample Value           |
| -------------------- | ------------------------ | ------- | ---------------------- |
| `ENABLE_OPENROUTER`  | Register OpenRouter models | Boolean | `true`, `false`        |
| `OPENROUTER_API_KEY` | OpenRouter API key        | String  | `sk-1234567890`        |
| `OPENROUTER_MODEL`   | OpenRouter model name     | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE`| OpenRouter API base URL | String | `https://api.openrouter.ai/v1`        |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                      | Description                                        | Type    | Sample Value                  |
| ----------------------------- | -------------------------------------------------- | ------- | ----------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`     | Register a custom OpenAI-compatible API endpoint    | Boolean | `true`, `false`               |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint        | String  | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc. |
| `OPENAI_COMPATIBLE_API_KEY`    | API key for OpenAI-compatible endpoint           | String  | `sk-1234567890`               |
| `OPENAI_COMPATIBLE_API_BASE`   | Base URL for OpenAI-compatible endpoint          | String  | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc. |
| `OPENAI_COMPATIBLE_API_VERSION`| API version for OpenAI-compatible endpoint, optional | String  | `2023-05-15`                  |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional           | Integer | `4096`, `8192`, etc.         |
| `OPENAI_COMPATIBLE_TEMPERATURE`| Temperature setting, optional                  | Float   | `0.0`, `0.5`, `0.7`, etc.     |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION`| Whether model supports vision, optional       | Boolean | `true`, `false`               |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable              | Description                                      | Type    | Sample Value        |
| --------------------- | ------------------------------------------------ | ------- | ------------------- |
| `LLM_KEY`            | The name of the model you want to use            | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY`  | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS`| Override the max tokens used by the LLM           | Integer | `128000`            |

## Feature Roadmap

*   \[x] **Open Source** - Open Source Skyvern's core codebase
*   \[x] **Workflow support** - Allow support to chain multiple Skyvern calls together
*   \[x] **Improved context** - Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
*   \[x] **Cost Savings** - Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
*   \[x] **Self-serve UI** - Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
*   \[x] **Workflow UI Builder** - Introduce a UI to allow users to build and analyze workflows visually
*   \[x] **Chrome Viewport streaming** - Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
*   \[x] **Past Runs UI** - Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
*   \[X] **Auto workflow builder ("Observer") mode** - Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
*   \[x] **Prompt Caching** - Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
*   \[x] **Web Evaluation Dataset** - Integrate Skyvern with public benchmark tests to track the quality of our models over time
*   \[ ] **Improved Debug mode** - Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
*   \[ ] **Chrome Extension** - Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
*   \[ ] **Skyvern Action Recorder** - Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
*   \[ ] **Interactable Livestream** - Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
*   \[ ] **Integrate LLM Observability tools** - Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
*   \[x] **Langchain Integration** - Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We welcome PRs and suggestions!  Please see our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started, and don't hesitate to open a PR/issue, or to reach out [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

By Default, Skyvern collects basic usage statistics to help us understand how Skyvern is being used. If you would like to opt-out of telemetry, please set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's open source repository is supported via a managed cloud. All of the core logic powering Skyvern is available in this open source repository licensed under the [AGPL-3.0 License](LICENSE), with the exception of anti-bot measures available in our managed cloud offering.

If you have any questions or concerns around licensing, please [contact us](mailto:support@skyvern.com) and we would be happy to help.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)