# Skyvern: Automate Browser Workflows with LLMs and Computer Vision

**Harness the power of Large Language Models (LLMs) and Computer Vision to automate complex browser-based tasks, replacing brittle automation solutions.**

[![Website](https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black)](https://www.skyvern.com/)
[![Docs](https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black)](https://docs.skyvern.com/)
[![Discord](https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord)](https://discord.gg/fG2XXEuQX3)
[![GitHub Stars](https://img.shields.io/github/stars/skyvern-ai/skyvern)](https://github.com/Skyvern-AI/skyvern)
[![License](https://img.shields.io/github/license/skyvern-ai/skyvern)](https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/skyvernai?style=social)](https://twitter.com/skyvernai)
[![LinkedIn](https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin)](https://www.linkedin.com/company/95726232)

[Skyvern](https://www.skyvern.com) empowers you to automate intricate browser-based workflows by leveraging the capabilities of LLMs and computer vision. Automate your tasks on any website, regardless of layout changes.

**Key Features:**

*   **LLM-Powered Automation:** Utilizes LLMs to understand and interact with websites, eliminating the need for brittle, code-dependent solutions.
*   **Resilient to Website Changes:**  Skyvern's vision-based approach is adaptable to evolving website layouts, ensuring automation continuity.
*   **Cross-Website Workflow Application:** Apply a single workflow across numerous websites, streamlining complex automation tasks.
*   **Advanced Reasoning:** LLMs enable Skyvern to reason through complex scenarios, such as inferring information from seemingly unrelated data.
*   **Real-time Monitoring and Debugging:**  Livestream the browser's viewport so you can see exactly what Skyvern is doing on the web.
*   **Form Filling Automation**: Skyvern natively fills out form inputs on websites, so it will comprehend the information and fill out forms accordingly.
*   **Data Extraction** Skyvern is also capable of extracting data from a website, just by specifying a data schema in the prompt.
*   **File Downloading**: Skyvern is capable of downloading files and automatically uploads them to block storage.
*   **Authentication Support**: Skyvern supports different authentication methods to automate tasks behind login walls.
*   **Integrations:** Supports MCP, Zapier, Make.com, and N8N to easily connect your workflows.

Want to see Skyvern in action? Check out the [Real-world examples](#real-world-examples-of-skyvern) section.

## Quickstart

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern

```bash
skyvern quickstart
```

### 3. Run task

#### UI (Recommended)

Start the Skyvern service and UI

```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern starts running the task in a browser that pops up and closes it when the task is done. You will be able to view the task from http://localhost:8080/history

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

Inspired by the task-driven autonomous agent design of BabyAGI and AutoGPT, Skyvern utilizes a swarm of agents powered by LLMs and Computer Vision to interact with websites using Playwright for browser automation.

Skyvern's architecture offers several advantages:

*   **Adaptability:** Operates on unfamiliar websites by mapping visual elements to actions.
*   **Robustness:** Resists website layout changes, relying on a vision-based approach.
*   **Scalability:** Executes a single workflow across multiple websites, simplifying automation.
*   **Intelligent Interactions:** LLMs handle complex scenarios such as inferring answers and identifying products across various platforms.

Find a detailed technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern demonstrates SOTA performance on the [WebBench benchmark](webbench.ai), achieving 64.4% accuracy. Read the technical report and evaluation [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

### Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

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

```bash
docker rm -f postgresql-container
```

    If you encounter any database related errors while using Docker to run Skyvern, check which Postgres container is running with `docker ps`.

## Skyvern Features

### Skyvern Tasks

Tasks are the core building blocks within Skyvern, representing individual instructions to navigate and interact with websites.

Each task requires a `url` and a `prompt`, and optionally a `data schema` for structured output and `error codes` for specific error handling.

### Skyvern Workflows

Workflows enable the chaining of multiple tasks to create cohesive units of work.

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

### Livestreaming

Monitor Skyvern's actions in real-time through browser viewport livestreaming, useful for debugging and intervention.

### Form Filling

Skyvern natively fills out form inputs on websites. Passing information via the `navigation_goal` will allow Skyvern to comprehend the information and fill out the form accordingly.

### Data Extraction

Extract specific data points from websites.  You can also define a `data_extraction_schema` in JSONC format directly within the prompt to get structured results.

### File Downloading

Automated file downloads with automatic uploads to block storage.

### Authentication

Support for various authentication methods to automate tasks behind logins, including:

*   🔐 2FA Support (TOTP):  QR-based, email, and SMS-based 2FA. Learn more [here](https://docs.skyvern.com/credentials/totp).
*   Password Manager Integrations: Currently supports Bitwarden.

## Real-world examples of Skyvern

Explore practical applications of Skyvern:

*   **Invoice Downloading on many different websites**: [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

*   **Automate the job application process** [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

*   **Automate materials procurement for a manufacturing company** [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

*   **Navigating to government websites to register accounts or fill out forms** [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

*   **Filling out random contact us forms** [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

*   **Retrieving insurance quotes from insurance providers in any language**  [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)  [💡 See it in action](https://app.skyvern.com/tasks/create/geico)

## Contributor Setup

For a complete local environment CLI Installation

```bash
pip install -e .
```

The following command sets up your development environment to use pre-commit (our commit hook handler)

```
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` in your browser to start using the UI.

    \*The Skyvern CLI supports Windows, WSL, macOS, and Linux environments.\*

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com). For any questions or issues, please reach out [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider      | Supported Models                                                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                                                                                                                    |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                                                               |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                                                          |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                                                      |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                                                                                                              |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)                                                                       |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)                                                                                         |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible))                 |

### Environment Variables

#### OpenAI

| Variable            | Description                               | Type      | Sample Value        |
| ------------------- | ----------------------------------------- | --------- | ------------------- |
| `ENABLE_OPENAI`     | Register OpenAI models                   | Boolean   | `true`, `false`     |
| `OPENAI_API_KEY`    | OpenAI API Key                            | String    | `sk-1234567890`     |
| `OPENAI_API_BASE`   | OpenAI API Base, optional               | String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID, optional      | String    | `your-org-id`       |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

#### Anthropic

| Variable            | Description                     | Type      | Sample Value        |
| ------------------- | ------------------------------- | --------- | ------------------- |
| `ENABLE_ANTHROPIC`    | Register Anthropic models        | Boolean   | `true`, `false`     |
| `ANTHROPIC_API_KEY`   | Anthropic API key              | String    | `sk-1234567890`     |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

#### Azure OpenAI

| Variable              | Description                   | Type      | Sample Value                        |
| --------------------- | ----------------------------- | --------- | ----------------------------------- |
| `ENABLE_AZURE`        | Register Azure OpenAI models  | Boolean   | `true`, `false`                     |
| `AZURE_API_KEY`       | Azure deployment API key      | String    | `sk-1234567890`                     |
| `AZURE_DEPLOYMENT`    | Azure OpenAI Deployment Name  | String    | `skyvern-deployment`                |
| `AZURE_API_BASE`      | Azure deployment api base url | String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`   | Azure API Version             | String    | `2024-02-01`                        |

Recommended `LLM_KEY`: `AZURE_OPENAI`

#### AWS Bedrock

| Variable             | Description                                                                                                                                           | Type      | Sample Value        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------------- |
| `ENABLE_BEDROCK`     | Register AWS Bedrock models. To use AWS Bedrock, you need to make sure your [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3) are set up correctly first. | Boolean   | `true`, `false`     |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

#### Gemini

| Variable         | Description                | Type      | Sample Value        |
| ---------------- | -------------------------- | --------- | ------------------- |
| `ENABLE_GEMINI`    | Register Gemini models     | Boolean   | `true`, `false`     |
| `GEMINI_API_KEY`   | Gemini API Key             | String    | `your_google_gemini_api_key`  |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

#### Ollama

| Variable          | Description                        | Type      | Sample Value                       |
| ----------------- | ---------------------------------- | --------- | ---------------------------------- |
| `ENABLE_OLLAMA`     | Register local models via Ollama  | Boolean   | `true`, `false`                    |
| `OLLAMA_SERVER_URL` | URL for your Ollama server        | String    | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`      | Ollama model name to load         | String    | `qwen2.5:7b-instruct`               |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

#### OpenRouter

| Variable          | Description                     | Type      | Sample Value        |
| ----------------- | ------------------------------- | --------- | ------------------- |
| `ENABLE_OPENROUTER` | Register OpenRouter models      | Boolean   | `true`, `false`     |
| `OPENROUTER_API_KEY` | OpenRouter API key            | String    | `sk-1234567890`     |
| `OPENROUTER_MODEL`  | OpenRouter model name           | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL         | String    | `https://api.openrouter.ai/v1`    |

Recommended `LLM_KEY`: `OPENROUTER`

#### OpenAI-Compatible

| Variable                       | Description                                   | Type      | Sample Value                         |
| ------------------------------ | --------------------------------------------- | --------- | ------------------------------------ |
| `ENABLE_OPENAI_COMPATIBLE`     | Register a custom OpenAI-compatible API endpoint | Boolean   | `true`, `false`                      |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint     | String    | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.  |
| `OPENAI_COMPATIBLE_API_KEY`    | API key for OpenAI-compatible endpoint        | String    | `sk-1234567890`                       |
| `OPENAI_COMPATIBLE_API_BASE`   | Base URL for OpenAI-compatible endpoint       | String    | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc. |
| `OPENAI_COMPATIBLE_API_VERSION`| API version for OpenAI-compatible endpoint, optional| String    | `2023-05-15`                         |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional       | Integer   | `4096`, `8192`, etc.                |
| `OPENAI_COMPATIBLE_TEMPERATURE`| Temperature setting, optional                 | Float     | `0.0`, `0.5`, `0.7`, etc.          |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional    | Boolean   | `true`, `false`                      |

Supported LLM Key: `OPENAI_COMPATIBLE`

#### General LLM Configuration

| Variable            | Description                            | Type    | Sample Value |
| ------------------- | -------------------------------------- | ------- | ------------ |
| `LLM_KEY`           | The name of the model you want to use  | String  | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | The name of the model for mini agents skyvern runs with | String  | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |

## Feature Roadmap

*   \[x] Open Source - Open Source Skyvern's core codebase
*   \[x] Workflow support - Allow support to chain multiple Skyvern calls together
*   \[x] Improved context - Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
*   \[x] Cost Savings - Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
*   \[x] Self-serve UI - Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
*   \[x] Workflow UI Builder - Introduce a UI to allow users to build and analyze workflows visually
*   \[x] Chrome Viewport streaming - Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
*   \[x] Past Runs UI - Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
*   \[X] Auto workflow builder ("Observer") mode - Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
*   \[x] Prompt Caching - Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
*   \[x] Web Evaluation Dataset - Integrate Skyvern with public benchmark tests to track the quality of our models over time
*   \[ ] Improved Debug mode - Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
*   \[ ] Chrome Extension - Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
*   \[ ] Skyvern Action Recorder - Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
*   \[ ] Interactable Livestream - Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
*   \[ ] Integrate LLM Observability tools - Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
*   \[x] Langchain Integration - Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We welcome contributions! Open a PR/issue, or contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).  See our [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for more.

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects usage statistics by default to help improve the product.  To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's core logic is open source under the [AGPL-3.0 License](LICENSE).  Managed cloud features have proprietary anti-bot measures.

For licensing questions, [contact us](mailto:support@skyvern.com).

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)