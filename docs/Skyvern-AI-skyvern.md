<h1 align="center">
 <a href="https://www.skyvern.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png"/>
  </picture>
 </a>
 <br />
</h1>

## Automate Any Browser Workflow with AI: Skyvern

Skyvern harnesses the power of Large Language Models (LLMs) and computer vision to automate browser-based workflows, offering a revolutionary approach to web automation.  **[Explore the Skyvern GitHub Repository](https://github.com/Skyvern-AI/skyvern) to get started.**

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

Skyvern eliminates the need for brittle, website-specific scripts by using advanced AI to interact with web pages. This allows you to automate complex tasks across numerous websites with ease.

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif"/>
</p>

**Key Features:**

*   **AI-Powered Automation:** Leverage LLMs and computer vision for intelligent web interaction.
*   **Website Agnostic:** Works on sites you've never seen before, adapting to layout changes.
*   **Scalable Automation:** Apply a single workflow to multiple websites efficiently.
*   **Intelligent Decision-Making:** LLMs enable smart reasoning for complex scenarios.
*   **Advanced Capabilities:**  Includes form filling, data extraction, file downloading, and authentication support.
*   **Workflow Builder:**  Allows chaining multiple tasks together to form a cohesive unit of work.

### How it Works

Skyvern employs a task-driven approach, similar to BabyAGI and AutoGPT, but with the ability to directly interact with websites using browser automation (Playwright).  Agents comprehend, plan, and execute actions, overcoming the limitations of traditional automation methods.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" />
</picture>

### Performance & Evaluation

Skyvern demonstrates state-of-the-art performance, particularly in WRITE tasks (e.g., form filling).

*   **WebBench Benchmark:** Skyvern achieved 64.4% accuracy.
*   **WRITE Task Leader:** Skyvern is the top-performing agent in WRITE tasks, essential for RPA.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

### Quickstart

#### Skyvern Cloud
[Skyvern Cloud](https://app.skyvern.com) is a managed cloud version of Skyvern that allows you to run Skyvern without worrying about the infrastructure.

#### Install & Run

**Prerequisites:**

*   [Python 3.11.x](https://www.python.org/downloads/)
*   [NodeJS & NPM](https://nodejs.org/en/download/)
*   (Windows only) [Rust](https://rustup.rs/), VS Code with C++ dev tools, and Windows SDK

```bash
pip install skyvern
```

```bash
skyvern quickstart
```

```bash
skyvern run all
```

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

### Advanced Usage

*   **Control Your Own Browser** (Chrome)
    *   Use Chrome by specifying the browser path in your code.
    *   or use the .env variables, `CHROME_EXECUTABLE_PATH` and `BROWSER_TYPE=cdp-connect`.
*   **Run with a Remote Browser**
    *   Use the cdp connection url in your code.
*   **Get Consistent Output Schema**
    *   Use `data_extraction_schema`.

### Docker Compose setup

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

### Examples
Explore real-world applications of Skyvern:

*   **Invoice Downloading:**  Automate invoice downloads from various websites.  [Book a demo](https://meetings.hubspot.com/skyvern/demo).
    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   **Job Application Automation:** Automate your job application process.  [See it in action](https://app.skyvern.com/tasks/create/job_application).
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>
*   **Material Procurement:** Automate material procurement for manufacturers. [See it in action](https://app.skyvern.com/tasks/create/finditparts).
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>

*   **Government Website Navigation:** Automate registration and form filling on government websites.  [See it in action](https://app.skyvern.com/tasks/create/california_edd).
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>

*   **Contact Form Automation:** Fill out contact us forms automatically. [See it in action](https://app.skyvern.com/tasks/create/contact_us_forms).
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>

*   **Insurance Quote Retrieval:** Get insurance quotes from providers in any language. [See it in action](https://app.skyvern.com/tasks/create/bci_seguros).
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>

    [See it in action](https://app.skyvern.com/tasks/create/geico)

    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

### Documentation

Comprehensive documentation is available on our [📕 docs page](https://docs.skyvern.com).

### Supported LLMs

| Provider      | Supported Models                                                                                                                                |
| :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                                                                                                   |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                                                             |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                                                        |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                                                    |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                                                                                           |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)                                                                      |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                                                                                        |
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

### Feature Roadmap
Explore planned enhancements:
*   [ ] Improved Debug mode
*   [ ] Chrome Extension
*   [ ] Skyvern Action Recorder
*   [ ] Interactable Livestream
*   [ ] Integrate LLM Observability tools
*   [x] Langchain Integration

### Contributing
We welcome contributions! See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).  Or reach out to us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

### Telemetry
Skyvern collects basic usage data. Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

### License
Skyvern is licensed under the [AGPL-3.0 License](LICENSE), except for anti-bot measures in the managed cloud. Contact [support@skyvern.com](mailto:support@skyvern.com) for licensing questions.

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)