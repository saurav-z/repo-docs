<h1 align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
        <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
    </picture>
    <br/>
    Skyvern: Automate Browser Workflows with AI
</h1>

<p align="center">
    🤖 Effortlessly automate web tasks using Large Language Models (LLMs) and Computer Vision with <a href="https://github.com/Skyvern-AI/skyvern">Skyvern</a>.
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

**Skyvern** empowers you to automate browser-based workflows by leveraging the power of LLMs and computer vision. This innovative approach replaces fragile, script-based automation with a more robust and adaptable solution.

<p align="center">
    <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo"/>
</p>

## Key Features

*   **AI-Powered Automation:** Automate tasks on websites you've never seen before, adapting to layout changes.
*   **Workflow Flexibility:** Chain multiple tasks to create complex automation sequences.
*   **Visual Understanding:** Leverage computer vision and LLMs to understand and interact with web elements.
*   **Data Extraction:** Extract specific data from websites using schema-based output.
*   **Form Filling:**  Automate form submissions.
*   **2FA Support:** Secure your workflows with 2FA (TOTP, QR-based, SMS and Email-based).
*   **Real-Time Monitoring:** Live-stream the browser's viewport for debugging and control.
*   **Integrations:**  Connect Skyvern with Zapier, Make.com, and N8N for enhanced automation.
*   **Password Manager Integrations:**  Seamlessly integrate with password managers such as Bitwarden.
*   **Model Context Protocol (MCP):** Use any LLM that supports MCP.
*   **Docker Compose support:** Quickly set up Skyvern using Docker Compose for easy deployment.

## How Skyvern Works

Skyvern, inspired by task-driven autonomous agent designs like BabyAGI and AutoGPT, uses a swarm of AI agents to:

1.  **Comprehend:** Understand a website's structure.
2.  **Plan:**  Devise the steps needed to complete a task.
3.  **Execute:** Perform actions using browser automation libraries like Playwright.

**Key Advantages:**

1.  **Adaptability:** Works on unfamiliar websites without custom code.
2.  **Resilience:**  Withstands website layout changes.
3.  **Scalability:**  Applies a single workflow to many different websites.
4.  **Intelligence:** LLMs reason through complex scenarios (e.g., insurance eligibility, product matching).

**Detailed technical report:** [https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/)

## Demo

[Demo Video](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

## Performance & Evaluation

Skyvern shows SOTA performance on the [WebBench benchmark](webbench.ai) with 64.4% accuracy, and the best performing agent on write tasks.

<p align="center">
    <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
</p>

<p align="center">
    <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Performance"/>
</p>

## Quickstart

### 1. Skyvern Cloud

Get started quickly with [Skyvern Cloud](https://app.skyvern.com), a managed service that eliminates infrastructure management.

### 2. Install & Run Locally

#### Prerequisites

*   [Python 3.11.x](https://www.python.org/downloads/) (compatible with 3.12, not yet with 3.13)
*   [NodeJS & NPM](https://nodejs.org/en/download/)

#### Additional for Windows

*   [Rust](https://rustup.rs/)
*   VS Code with C++ dev tools and Windows SDK

#### Installation

```bash
pip install skyvern
```

#### Run

```bash
skyvern quickstart
```

### 3. Run Tasks

#### UI (Recommended)

```bash
skyvern run all
```

Access the UI at `http://localhost:8080` to run tasks.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

More examples of running tasks with different targets.

## Advanced Usage

### Control Your Browser (Chrome)

1.  **With Python Code:**

    ```python
    from skyvern import Skyvern

    browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" #Example for Mac
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
        CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" # Example for Mac
        BROWSER_TYPE=cdp-connect
        ```

    *   Restart the Skyvern service: `skyvern run all`

### Run with any Remote Browser

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

### Debugging Commands

```bash
skyvern run server # Launch Skyvern Server Separately
skyvern run ui # Launch Skyvern UI
skyvern status # Check status of Skyvern service
skyvern stop all # Stop the Skyvern service
skyvern stop ui # Stop Skyvern UI
skyvern stop server # Stop Skyvern Server Separately
```

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Make sure you don't have postgres running locally (Run `docker ps` to check).
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

## Skyvern Features - Detailed

### Skyvern Tasks

Tasks are the foundation of Skyvern: single requests to automate a web-based goal. They require `url`, `prompt`, and can optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Tasks Screenshot"/>
</p>

### Skyvern Workflows

Build cohesive automation by chaining multiple tasks:

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
  <img src="fern/images/block_example_v2.png" alt="Workflow Example"/>
</p>

### Livestreaming

Watch Skyvern interact with the web in real-time for debugging and insight.

### Form Filling

Skyvern excels at filling out web forms via the `navigation_goal`.

### Data Extraction

Extract data from websites, formatted according to a specified `data_extraction_schema`.

### File Downloading

Download files from the web, automatically stored in block storage (if configured).

### Authentication

Support for various authentication methods:

*   QR-based 2FA (Google Authenticator, Authy, etc.)
*   Email-based 2FA
*   SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

Currently supports:

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Use any LLM that supports the Model Context Protocol.  See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Integrations

Connect Skyvern to other apps with:

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

## Real-World Examples

See Skyvern in action:

### Invoice Downloading

[Book a demo](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo"/>
</p>

### Automate Job Applications

[See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Demo"/>
</p>

### Procurement Automation

[See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Procurement Automation"/>
</p>

### Government Website Navigation

[See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="Government Website Navigation"/>
</p>

### Contact Us Form Filling

[See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Us Form Filling"/>
</p>

### Insurance Quote Retrieval

[See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Retrieval Example 1"/>
</p>

[See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Insurance Quote Retrieval Example 2"/>
</p>

## Documentation

Comprehensive documentation is available on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

| Provider      | Supported Models                                           |
| ------------- | ---------------------------------------------------------- |
| OpenAI        | gpt4-turbo, gpt-4o, gpt-4o-mini                           |
| Anthropic     | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)      |
| Azure OpenAI  | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock   | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Gemini        | Gemini 2.5 Pro and flash, Gemini 2.0                     |
| Ollama        | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter    | Access models through [OpenRouter](https://openrouter.ai)   |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

### Environment Variables - Configuration

#### OpenAI

| Variable            | Description               | Type      | Sample Value           |
| ------------------- | ------------------------- | --------- | ----------------------- |
| `ENABLE_OPENAI`     | Enable OpenAI models      | Boolean   | `true`, `false`         |
| `OPENAI_API_KEY`    | OpenAI API Key            | String    | `sk-1234567890`         |
| `OPENAI_API_BASE`   | OpenAI API Base (optional)| String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID (optional) | String | `your-org-id` |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

#### Anthropic

| Variable             | Description            | Type      | Sample Value        |
| -------------------- | ---------------------- | --------- | ------------------- |
| `ENABLE_ANTHROPIC`   | Enable Anthropic models| Boolean   | `true`, `false`     |
| `ANTHROPIC_API_KEY`  | Anthropic API key      | String    | `sk-1234567890`     |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

#### Azure OpenAI

| Variable          | Description                  | Type      | Sample Value                        |
| ----------------- | ---------------------------- | --------- | ----------------------------------- |
| `ENABLE_AZURE`    | Enable Azure OpenAI models   | Boolean   | `true`, `false`                    |
| `AZURE_API_KEY`   | Azure deployment API key     | String    | `sk-1234567890`                    |
| `AZURE_DEPLOYMENT`| Azure OpenAI Deployment Name | String    | `skyvern-deployment`               |
| `AZURE_API_BASE`  | Azure deployment api base url| String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION`| Azure API Version           | String    | `2024-02-01`                        |

Recommended `LLM_KEY`: `AZURE_OPENAI`

#### AWS Bedrock

| Variable            | Description                     | Type      | Sample Value        |
| ------------------- | ------------------------------- | --------- | ------------------- |
| `ENABLE_BEDROCK`    | Enable AWS Bedrock models. (Requires proper [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3))   | Boolean   | `true`, `false`     |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

#### Gemini

| Variable         | Description      | Type      | Sample Value                   |
| ---------------- | ---------------- | --------- | ------------------------------ |
| `ENABLE_GEMINI`  | Enable Gemini models | Boolean   | `true`, `false`                |
| `GEMINI_API_KEY` | Gemini API Key   | String    | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

#### Ollama

| Variable           | Description             | Type      | Sample Value                  |
| ------------------ | ----------------------- | --------- | ----------------------------- |
| `ENABLE_OLLAMA`    | Enable Ollama models    | Boolean   | `true`, `false`               |
| `OLLAMA_SERVER_URL` | Ollama server URL       | String    | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`      | Ollama model name      | String    | `qwen2.5:7b-instruct`         |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

#### OpenRouter

| Variable             | Description         | Type      | Sample Value                      |
| -------------------- | ------------------- | --------- | --------------------------------- |
| `ENABLE_OPENROUTER`  | Enable OpenRouter models | Boolean   | `true`, `false`                   |
| `OPENROUTER_API_KEY` | OpenRouter API key | String    | `sk-1234567890`                   |
| `OPENROUTER_MODEL`   | OpenRouter model name | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | String    | `https://api.openrouter.ai/v1`    |

Recommended `LLM_KEY`: `OPENROUTER`

#### OpenAI-Compatible

| Variable                       | Description                               | Type      | Sample Value                            |
| ------------------------------ | ----------------------------------------- | --------- | --------------------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`     | Enable custom OpenAI-compatible endpoint | Boolean   | `true`, `false`                         |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for endpoint                | String    | `yi-34b`, `gpt-3.5-turbo`, etc.         |
| `OPENAI_COMPATIBLE_API_KEY`    | API key for endpoint                    | String    | `sk-1234567890`                         |
| `OPENAI_COMPATIBLE_API_BASE`   | Base URL for endpoint                   | String    | `https://api.together.xyz/v1`, etc.    |
| `OPENAI_COMPATIBLE_API_VERSION`| API version (optional)                   | String    | `2023-05-15`                            |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens (optional)                | Integer   | `4096`, `8192`, etc.                    |
| `OPENAI_COMPATIBLE_TEMPERATURE`| Temperature setting (optional)             | Float     | `0.0`, `0.5`, `0.7`, etc.               |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION`| Whether model supports vision (optional)            | Boolean     | `true`, `false`                         |

Supported LLM Key: `OPENAI_COMPATIBLE`

#### General LLM Configuration

| Variable             | Description                    | Type      | Sample Value      |
| -------------------- | ------------------------------ | --------- | ----------------- |
| `LLM_KEY`            | Main LLM model to use          | String    | See supported keys |
| `SECONDARY_LLM_KEY`  | Mini agent LLM                 | String    | See supported keys |
| `LLM_CONFIG_MAX_TOKENS` | Override max tokens for LLM    | Integer   | `128000`          |

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

Contributions are welcome! Please review the [contribution guide](CONTRIBUTING.md) and "Help Wanted" issues to get started: [https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage data by default. Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

This open-source project is licensed under the [AGPL-3.0 License](LICENSE), with anti-bot measures reserved for the managed cloud offering. Contact us at [support@skyvern.com](mailto:support@skyvern.com) for licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)