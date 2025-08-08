<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
  </picture>
  <br>
  Skyvern: Automate Browser Workflows with the Power of LLMs and Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="Stars"></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"></a>
</p>

Skyvern empowers you to automate complex browser-based tasks by leveraging Large Language Models (LLMs) and computer vision, enabling you to automate repetitive tasks on websites effortlessly. **Visit the [original repository](https://github.com/Skyvern-AI/skyvern) for more details.**

## Key Features

*   🐉 **LLM-Powered Automation:** Automates browser interactions using the power of LLMs, enabling you to automate a wide range of tasks without relying on brittle code.
*   👁️ **Computer Vision Integration:** Skyvern visually understands and interacts with web pages, adapting to layout changes without requiring code modifications.
*   💻 **Flexible Workflow Creation:** Create intricate workflows that combine navigation, data extraction, form filling, file handling, and integrations.
*   🚀 **Cloud and Local Deployment:** Offers both a managed cloud version ([Skyvern Cloud](https://app.skyvern.com)) and a local installation, providing flexibility for different use cases.
*   🌐 **Cross-Website Compatibility:**  Works across numerous websites with minimal configuration, making it a versatile solution for various automation needs.
*   🔑 **Authentication Support:** Supports various authentication methods, including 2FA, to automate tasks behind logins securely.
*   🔄 **Real-time Monitoring and Debugging:** Provides live browser viewport streaming for debugging and understanding how Skyvern interacts with websites.
*   🔗 **Extensive Integrations:** Integrates with tools like Zapier, Make.com, and N8N, expanding the possibilities of automation.
*   📊 **Data Extraction Capabilities:**  Extracts structured data from web pages using schema definitions, ensuring output consistency.
*   📦 **File Handling:** Downloads and automatically uploads files to block storage.

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

Visit `http://localhost:8080` to use the UI and run a task.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

You can also run tasks using different configurations, such as Skyvern Cloud or a local service:

```python
from skyvern import Skyvern

# Run on Skyvern Cloud
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Local Skyvern service
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## How it Works

Inspired by autonomous agent designs like BabyAGI and AutoGPT, Skyvern incorporates the ability to interact with websites using browser automation libraries like Playwright.

Skyvern uses a swarm of agents to comprehend a website, and plan and execute its actions:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" />
</picture>

Key advantages include:

*   **Website Agnostic:** Operates on websites it hasn't seen before.
*   **Layout Change Resistant:** Adaptable to website layout changes without code modifications.
*   **Scalable Workflow Application:** Applies workflows across numerous websites.
*   **LLM Reasoning:** Uses LLMs for complex situations, enhancing the flexibility of the tasks.

A detailed technical report can be found [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<!-- Redo demo -->
https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern demonstrates SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. The technical report + evaluation can be found [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

### Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Performance"/>
</p>

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
2.  Make sure you don't have postgres running locally (Run `docker ps` to check)
3.  Clone the repository and navigate to the root directory
4.  Run `skyvern init llm` to generate a `.env` file. This will be copied into the Docker image.
5.  Fill in the LLM provider key on the [docker-compose.yml](./docker-compose.yml). *If you want to run Skyvern on a remote server, make sure you set the correct server ip for the UI container in [docker-compose.yml](./docker-compose.yml).*
6.  Run the following command via the commandline:
    ```bash
     docker compose up -d
    ```
7.  Navigate to `http://localhost:8080` in your browser to start using the UI

>   **Important:** Only one Postgres container can run on port 5432 at a time. If you switch from the CLI-managed Postgres to Docker Compose, you must first remove the original container:
>   ```bash
>   docker rm -f postgresql-container
>   ```

If you encounter any database related errors while using Docker to run Skyvern, check which Postgres container is running with `docker ps`.

## Skyvern Features

### Skyvern Tasks

Tasks are the fundamental building blocks within Skyvern. Each task represents a single request, instructing Skyvern to navigate a website and achieve a specified goal.

Tasks require a `url`, `prompt`, and can include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Example">
</p>

### Skyvern Workflows

Workflows allow you to chain multiple tasks together to perform more complex operations.

For instance, you could create a workflow to download invoices, automate e-commerce purchases, or other multi-step processes.

Supported features include:

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

Skyvern enables you to livestream the browser viewport to your local machine, aiding in debugging and task understanding.

### Form Filling

Skyvern natively supports filling out form inputs on websites.

### Data Extraction

Skyvern is capable of extracting data from a website.  You can also specify a `data_extraction_schema` within the main prompt.

### File Downloading

Skyvern can download files from websites, automatically uploading them to block storage.

### Authentication

Skyvern supports multiple authentication methods. Contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3) to learn more.

<p align="center">
  <img src="fern/images/secure_password_task_example.png" alt="Authentication Example">
</p>

#### 🔐 2FA Support (TOTP)

Skyvern supports various 2FA methods:

1.  QR-based 2FA (e.g. Google Authenticator, Authy)
2.  Email-based 2FA
3.  SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

Skyvern supports integrations with:

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Skyvern supports the Model Context Protocol (MCP) to use any LLM that supports MCP.

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Skyvern integrates with Zapier, Make.com, and N8N for easy workflow connections.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world examples of Skyvern

Here are some use cases of Skyvern:

### Invoice Downloading on many different websites

[Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Example">
</p>

### Automate the job application process

[💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Example">
</p>

### Automate materials procurement for a manufacturing company

[💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Finditparts Example">
</p>

### Navigating to government websites to register accounts or fill out forms

[💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="California EDD Example">
</p>

### Filling out random contact us forms

[💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Forms Example">
</p>

### Retrieving insurance quotes from insurance providers in any language

[💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="BCI Seguros Example">
</p>

[💡 See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Example">
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

Find more extensive documentation on our [📕 docs page](https://docs.skyvern.com). Contact us with any questions [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                                    |
|---------------|--------------------------------------------------------------------|
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                 |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)              |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)              |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                               |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)       |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                   |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible))           |

#### Environment Variables

##### OpenAI

| Variable             | Description                                           | Type      | Sample Value       |
| -------------------- | ----------------------------------------------------- | --------- | ------------------ |
| `ENABLE_OPENAI`      | Register OpenAI models                                | Boolean   | `true`, `false`    |
| `OPENAI_API_KEY`     | OpenAI API Key                                        | String    | `sk-1234567890`    |
| `OPENAI_API_BASE`    | OpenAI API Base, optional                             | String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION`| OpenAI Organization ID, optional                      | String    | `your-org-id`      |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable             | Description                         | Type      | Sample Value       |
| -------------------- | ----------------------------------- | --------- | ------------------ |
| `ENABLE_ANTHROPIC`   | Register Anthropic models           | Boolean   | `true`, `false`    |
| `ANTHROPIC_API_KEY`  | Anthropic API key                   | String    | `sk-1234567890`    |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable             | Description                           | Type      | Sample Value            |
| -------------------- | ------------------------------------- | --------- | ----------------------- |
| `ENABLE_AZURE`       | Register Azure OpenAI models          | Boolean   | `true`, `false`         |
| `AZURE_API_KEY`      | Azure deployment API key              | String    | `sk-1234567890`         |
| `AZURE_DEPLOYMENT`   | Azure OpenAI Deployment Name          | String    | `skyvern-deployment`    |
| `AZURE_API_BASE`     | Azure deployment api base url         | String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`  | Azure API Version                     | String    | `2024-02-01`            |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable             | Description                                                                                                          | Type      | Sample Value       |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- | --------- | ------------------ |
| `ENABLE_BEDROCK`     | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean   | `true`, `false`    |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable         | Description                 | Type      | Sample Value              |
| ---------------- | --------------------------- | --------- | ------------------------- |
| `ENABLE_GEMINI`  | Register Gemini models      | Boolean   | `true`, `false`           |
| `GEMINI_API_KEY` | Gemini API Key              | String    | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable             | Description                               | Type      | Sample Value                     |
| -------------------- | ----------------------------------------- | --------- | -------------------------------- |
| `ENABLE_OLLAMA`      | Register local models via Ollama          | Boolean   | `true`, `false`                  |
| `OLLAMA_SERVER_URL`  | URL for your Ollama server                | String    | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`       | Ollama model name to load               | String    | `qwen2.5:7b-instruct`            |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable             | Description                   | Type      | Sample Value              |
| -------------------- | ----------------------------- | --------- | ------------------------- |
| `ENABLE_OPENROUTER`  | Register OpenRouter models      | Boolean   | `true`, `false`           |
| `OPENROUTER_API_KEY` | OpenRouter API key            | String    | `sk-1234567890`           |
| `OPENROUTER_MODEL`   | OpenRouter model name         | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL       | String    | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                       | Description                                   | Type      | Sample Value              |
| ------------------------------ | --------------------------------------------- | --------- | ------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`     | Register a custom OpenAI-compatible API endpoint| Boolean   | `true`, `false`           |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint    | String    | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY`    | API key for OpenAI-compatible endpoint       | String    | `sk-1234567890`           |
| `OPENAI_COMPATIBLE_API_BASE`   | Base URL for OpenAI-compatible endpoint      | String    | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION`| API version for OpenAI-compatible endpoint, optional| String  | `2023-05-15`           |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional    | Integer   | `4096`, `8192`, etc.      |
| `OPENAI_COMPATIBLE_TEMPERATURE`| Temperature setting, optional               | Float     | `0.0`, `0.5`, `0.7`, etc. |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION`| Whether model supports vision, optional   | Boolean   | `true`, `false`           |

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable                  | Description                           | Type    | Sample Value          |
| ------------------------- | ------------------------------------- | ------- | --------------------- |
| `LLM_KEY`                 | The name of the model you want to use  | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY`       | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS`   | Override the max tokens used by the LLM | Integer | `128000`             |

## Feature Roadmap

*   [x] **Open Source** - Open Source Skyvern's core codebase
*   [x] **Workflow support** - Allow support to chain multiple Skyvern calls together
*   [x] **Improved context** - Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
*   [x] **Cost Savings** - Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
*   [x] **Self-serve UI** - Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
*   [x] **Workflow UI Builder** - Introduce a UI to allow users to build and analyze workflows visually
*   [x] **Chrome Viewport streaming** - Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
*   [x] **Past Runs UI** - Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
*   [X] **Auto workflow builder ("Observer") mode** - Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
*   [x] **Prompt Caching** - Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
*   [x] **Web Evaluation Dataset** - Integrate Skyvern with public benchmark tests to track the quality of our models over time
*   [ ] **Improved Debug mode** - Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
*   [ ] **Chrome Extension** - Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
*   [ ] **Skyvern Action Recorder** - Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
*   [ ] **Interactable Livestream** - Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
*   [ ] **Integrate LLM Observability tools** - Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
*   [x] **Langchain Integration** - Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We welcome contributions! Open a PR/issue or reach out [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3). Check our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

By Default, Skyvern collects basic usage statistics. Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE). Anti-bot measures are available in our managed cloud offering. For licensing questions, please [contact us](mailto:support@skyvern.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)