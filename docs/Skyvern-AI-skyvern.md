<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
    </picture>
  </a>
  <br>
  <br>
  <p>
    <a href="https://github.com/Skyvern-AI/skyvern">
      <img src="https://img.shields.io/github/stars/skyvern-ai/skyvern?style=social" alt="GitHub stars">
    </a>
  </p>
</h1>

## Automate Browser Workflows with AI: Introducing Skyvern 🐉

Skyvern is a powerful open-source tool that automates browser-based workflows using Large Language Models (LLMs) and computer vision, enabling you to automate complex tasks across various websites. 

**Key Features:**

*   ✅ **AI-Powered Automation:** Automate repetitive tasks by simply describing what you want to do.
*   ✅ **Website Agnostic:** Works on websites it's never seen before.
*   ✅ **Resilient to Layout Changes:** Adapts to website updates without requiring code changes.
*   ✅ **Workflow Builder:** Chain tasks for complex automation scenarios.
*   ✅ **Data Extraction:** Easily extract structured data from websites.
*   ✅ **Advanced Features:** Includes Livestreaming, Form Filling, Authentication, 2FA Support, Password Manager integrations, and integrations with popular tools like Zapier, Make.com and N8N.
*   ✅ **Open Source**: Core logic is available under the AGPL-3.0 License

> Automate your web-based tasks today! Check out the [Skyvern GitHub Repo](https://github.com/Skyvern-AI/skyvern) to get started.

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Demo">
</p>

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

#### UI (Recommended)

```bash
skyvern run all
```

Visit http://localhost:8080 to run a task through the UI.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## How Skyvern Works

Skyvern uses a task-driven agent design, similar to BabyAGI and AutoGPT, but with the ability to interact with websites using libraries like Playwright. It uses a swarm of agents to understand, plan, and execute actions on websites.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png">
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram">
</picture>

This approach offers several advantages, including the ability to operate on unseen websites, resistance to layout changes, and applying workflows across a multitude of sites. Skyvern leverages LLMs for complex reasoning, like inferring answers based on context.

A detailed technical report is available [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern demonstrates SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. Find the technical report and evaluation [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Performance">
</p>

### Performance on WRITE tasks

Skyvern excels in WRITE tasks, commonly used for RPA applications.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Performance">
</p>

## Advanced Usage

### Control Your Own Browser (Chrome)

> ⚠️ WARNING: Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1.  **With Python Code**

```python
from skyvern import Skyvern

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

Add these variables to your `.env` file:

```bash
CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BROWSER_TYPE=cdp-connect
```

Restart Skyvern service with `skyvern run all`.

### Run with a Remote Browser

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Consistent Output Schema

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

### Helpful Commands

```bash
skyvern run server      # Launch the Skyvern Server Separately
skyvern run ui          # Launch the Skyvern UI
skyvern status          # Check Skyvern service status
skyvern stop all        # Stop the Skyvern service
skyvern stop ui         # Stop the Skyvern UI
skyvern stop server     # Stop the Skyvern Server Separately
```

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Ensure no local Postgres instance is running (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to generate a `.env` file.
5.  Populate the LLM provider key in `docker-compose.yml`.
6.  Run `docker compose up -d`.
7.  Access the UI at `http://localhost:8080`.

> **Important:** Remove the original Postgres container before switching from the CLI-managed Postgres to Docker Compose.

## Skyvern Features

### Skyvern Tasks

Tasks are the base unit, each representing a specific instruction. They require a `url`, `prompt`, and can optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Tasks">
</p>

### Skyvern Workflows

Workflows combine multiple tasks for cohesive processes.

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
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Workflow Example">
</p>

### Livestreaming

Real-time browser viewport streaming for debugging.

### Form Filling

Native form-filling capabilities via `navigation_goal`.

### Data Extraction

Extract data from websites, with schema support in JSONC format.

### File Downloading

Download files, automatically upload them to block storage.

### Authentication

Supports various authentication methods. Contact us for more information.

<p align="center">
  <img src="fern/images/secure_password_task_example.png" alt="Authentication Demo">
</p>

#### 🔐 2FA Support (TOTP)

Supports multiple 2FA methods, including QR-based, email, and SMS.

Learn more [here](https://docs.skyvern.com/credentials/totp).

#### Password Manager Integrations

Currently supports:

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Supports Model Context Protocol (MCP) for any LLM that supports it.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md).

### Zapier / Make.com / N8N Integration

Integrate Skyvern with these tools to connect to other apps.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

Learn more [here](https://docs.skyvern.com/credentials/totp).

## Real-world Examples

See Skyvern in action! Here are some examples of how Skyvern is being used:

### Invoice Downloading

[Book a demo](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading">
</p>

### Job Application Automation

[See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Demo">
</p>

### Materials Procurement

[See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement">
</p>

### Government Website Navigation

[See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="Government Website Navigation">
</p>

### Contact Us Form Filling

[See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Us Forms">
</p>

### Insurance Quote Retrieval

[See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Example">
</p>

[See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Example">
</p>

## Contributor Setup

```bash
pip install -e .
```

Set up your development environment:

```bash
skyvern quickstart contributors
```

Then:

1.  Navigate to `http://localhost:8080` in your browser.

   *The Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Find more documentation on our [📕 docs page](https://docs.skyvern.com). Contact us with any questions or suggestions [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                                                  |
| ------------- | --------------------------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                                   |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                             |
| Azure OpenAI  | Any GPT models (better performance with multimodal LLMs, e.g., azure/gpt4-o)      |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                     |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                              |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)       |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                        |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable              | Description            | Type    | Sample Value          |
| --------------------- | ---------------------- | ------- | --------------------- |
| `ENABLE_OPENAI`       | Register OpenAI models | Boolean | `true`, `false`       |
| `OPENAI_API_KEY`      | OpenAI API Key         | String  | `sk-1234567890`       |
| `OPENAI_API_BASE`     | OpenAI API Base        | String  | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID | String  | `your-org-id`         |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable              | Description         | Type    | Sample Value          |
| --------------------- | ------------------- | ------- | --------------------- |
| `ENABLE_ANTHROPIC`    | Register Anthropic  | Boolean | `true`, `false`       |
| `ANTHROPIC_API_KEY`   | Anthropic API key   | String  | `sk-1234567890`       |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable             | Description              | Type    | Sample Value              |
| -------------------- | ------------------------ | ------- | ------------------------- |
| `ENABLE_AZURE`       | Register Azure OpenAI   | Boolean | `true`, `false`           |
| `AZURE_API_KEY`      | Azure deployment API key | String  | `sk-1234567890`           |
| `AZURE_DEPLOYMENT`   | Deployment Name        | String  | `skyvern-deployment`     |
| `AZURE_API_BASE`     | Azure api base url      | String  | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`  | API Version              | String  | `2024-02-01`              |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable             | Description              | Type    | Sample Value              |
| -------------------- | ------------------------ | ------- | ------------------------- |
| `ENABLE_BEDROCK`     | Register AWS Bedrock     | Boolean | `true`, `false`           |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable           | Description         | Type    | Sample Value             |
| ------------------ | ------------------- | ------- | ------------------------ |
| `ENABLE_GEMINI`    | Register Gemini     | Boolean | `true`, `false`          |
| `GEMINI_API_KEY`   | Gemini API Key      | String  | `your_google_gemini_api_key`|

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable             | Description              | Type    | Sample Value                |
| -------------------- | ------------------------ | ------- | --------------------------- |
| `ENABLE_OLLAMA`      | Register local Ollama    | Boolean | `true`, `false`            |
| `OLLAMA_SERVER_URL`  | Ollama server URL       | String  | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`       | Ollama model name        | String  | `qwen2.5:7b-instruct`      |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable             | Description              | Type    | Sample Value                 |
| -------------------- | ------------------------ | ------- | ---------------------------- |
| `ENABLE_OPENROUTER`  | Register OpenRouter      | Boolean | `true`, `false`            |
| `OPENROUTER_API_KEY` | OpenRouter API key       | String  | `sk-1234567890`            |
| `OPENROUTER_MODEL`   | OpenRouter model name    | String  | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | String | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                          | Description                        | Type    | Sample Value                      |
| --------------------------------- | ---------------------------------- | ------- | --------------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`        | Register a custom endpoint         | Boolean | `true`, `false`                  |
| `OPENAI_COMPATIBLE_MODEL_NAME`    | Model name                         | String  | `yi-34b`, `gpt-3.5-turbo`, etc.  |
| `OPENAI_COMPATIBLE_API_KEY`       | API key                            | String  | `sk-1234567890`                  |
| `OPENAI_COMPATIBLE_API_BASE`      | Base URL                           | String  | `https://api.together.xyz/v1`     |
| `OPENAI_COMPATIBLE_API_VERSION`   | API version (optional)            | String  | `2023-05-15`                     |
| `OPENAI_COMPATIBLE_MAX_TOKENS`    | Max tokens (optional)             | Integer | `4096`, `8192`                   |
| `OPENAI_COMPATIBLE_TEMPERATURE`   | Temperature (optional)            | Float   | `0.0`, `0.5`, `0.7`              |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Supports vision  (optional)      | Boolean   | `true`, `false`                     |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable              | Description                     | Type    | Sample Value |
| --------------------- | ------------------------------- | ------- | ------------ |
| `LLM_KEY`             | Model to use                   | String  | See above     |
| `SECONDARY_LLM_KEY`   | Model for mini agents        | String  | See above     |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used | Integer | `128000`     |

## Feature Roadmap

- [x] **Open Source** - Open Source Skyvern's core codebase
- [x] **Workflow support** - Allow support to chain multiple Skyvern calls together
- [x] **Improved context** - Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
- [x] **Cost Savings** - Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
- [x] **Self-serve UI** - Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
- [x] **Workflow UI Builder** - Introduce a UI to allow users to build and analyze workflows visually
- [x] **Chrome Viewport streaming** - Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
- [x] **Past Runs UI** - Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
- [X] **Auto workflow builder ("Observer") mode** - Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
- [x] **Prompt Caching** - Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
- [x] **Web Evaluation Dataset** - Integrate Skyvern with public benchmark tests to track the quality of our models over time
- [ ] **Improved Debug mode** - Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
- [ ] **Chrome Extension** - Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
- [ ] **Skyvern Action Recorder** - Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
- [ ] **Interactable Livestream** - Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
- [ ] **Integrate LLM Observability tools** - Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
- [x] **Langchain Integration** - Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We welcome contributions! Open a PR/issue or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).
See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for help.

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics by default. To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern is open-source under the [AGPL-3.0 License](LICENSE), except for anti-bot measures in our managed cloud.

Contact us at [support@skyvern.com](mailto:support@skyvern.com) for licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)