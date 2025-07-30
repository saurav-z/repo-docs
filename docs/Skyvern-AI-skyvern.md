<!-- DOCTOC SKIP -->

<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
    </picture>
  </a>
  <br>
  Automate Your Browser-Based Workflows with the Power of LLMs and Computer Vision
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"></a>
</p>

[Skyvern](https://github.com/Skyvern-AI/skyvern) is an AI-powered browser automation tool that uses Large Language Models (LLMs) and computer vision to automate complex web workflows.  Replace manual, error-prone processes with intelligent automation, enabling you to interact with a wide variety of websites without the need for brittle, website-specific scripts.

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo">
</p>

**Key Features:**

*   **Intelligent Automation:** Uses LLMs and computer vision to understand and interact with websites, eliminating the need for custom scripts.
*   **Website Agnostic:** Works on websites you've never seen before, adapting to layout changes without requiring code modifications.
*   **Workflow Automation:** Chain multiple tasks together to create comprehensive automation workflows, from form filling to data extraction.
*   **Data Extraction:** Easily extract structured data from websites using customizable schema definitions.
*   **Livestreaming:** Real-time browser viewport streaming for debugging and monitoring.
*   **2FA Support:** Supports various 2FA methods, including QR-based, email, and SMS-based authentication.
*   **Integrations:** Seamlessly integrate with Zapier, Make.com, and N8N for powerful workflow connections.
*   **Self-hosted & Cloud Options:** Run Skyvern locally or in the cloud via [Skyvern Cloud](https://app.skyvern.com).
*   **Supports Open Source:** Leverages Open Source code and provides a platform to build on.

Want to see Skyvern in action?  Explore the [Real-world examples of Skyvern](#real-world-examples-of-skyvern) section below.

## Quickstart

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern

```bash
skyvern quickstart
```

### 3. Run a task

#### UI (Recommended)

Start the Skyvern service and UI:

```bash
skyvern run all
```

Go to [http://localhost:8080](http://localhost:8080) and use the UI to run a task.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern launches a browser, executes the task, and closes the browser when the task is complete.  View task history at [http://localhost:8080/history](http://localhost:8080/history).

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

Inspired by task-driven autonomous agent designs like BabyAGI and AutoGPT, Skyvern leverages LLMs and computer vision, but with the added ability to interact with websites using browser automation libraries such as Playwright.

Skyvern employs a swarm of intelligent agents to understand a website, plan actions, and execute them:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png">
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram">
</picture>

**Key Advantages:**

*   **Adaptability:** Operates on unseen websites by mapping visual elements to necessary actions.
*   **Resilience:** Resistant to website layout changes, relying on vision instead of rigid selectors.
*   **Scalability:** Applies a single workflow across multiple websites with intelligent reasoning.
*   **Intelligent Reasoning:** Uses LLMs to infer context and make complex decisions.

Find more details in the technical report: [Skyvern 2.0 Technical Report](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo
<!-- Redo demo -->
https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.  See the technical report and evaluation: [WebBench Evaluation](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance">
</p>

### Performance on WRITE Tasks (e.g., Filling Forms)

Skyvern excels in WRITE tasks (form filling, logins, downloads), key for Robotic Process Automation (RPA):

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Task Performance">
</p>

## Advanced Usage

### Control Your Browser (Chrome)

> ⚠️ **WARNING:** Chrome 136 and later disable default user data directory connections. Skyvern copies your user_data_dir to `./tmp/user_data_dir` on first connection. ⚠️

1.  **Python Code:**

```python
from skyvern import Skyvern

browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" # Example for Mac
skyvern = Skyvern(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY",
    browser_path=browser_path,
)
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

2.  **Skyvern Service (using .env file):**

Add these variables to your `.env` file:

```bash
CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # Example
BROWSER_TYPE=cdp-connect
```

Restart the Skyvern service: `skyvern run all`.  Then, run tasks through the UI or code.

### Run with a Remote Browser

Get the CDP connection URL and provide it to Skyvern:

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Consistent Output Schema

Specify the `data_extraction_schema` parameter:

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
# Launch Skyvern Server Separately
skyvern run server

# Launch the Skyvern UI
skyvern run ui

# Check Skyvern service status
skyvern status

# Stop the Skyvern service
skyvern stop all

# Stop the Skyvern UI
skyvern stop ui

# Stop the Skyvern Server
skyvern stop server
```

## Docker Compose Setup

1.  Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.
2.  Confirm that no local PostgreSQL instance is running (check with `docker ps`).
3.  Clone the repository and navigate to the root directory.
4.  Run `skyvern init llm` to create a `.env` file (copied into the Docker image).
5.  Enter your LLM provider key into [docker-compose.yml](./docker-compose.yml). *If using a remote server, configure the UI container's server IP in [docker-compose.yml](./docker-compose.yml).*
6.  Run the following command:
    ```bash
    docker compose up -d
    ```
7.  Access the UI at [http://localhost:8080](http://localhost:8080).

> **Important:**  Only one PostgreSQL container can run on port 5432.  Remove the original container if switching from the CLI: `docker rm -f postgresql-container`.

If database errors occur, check the running PostgreSQL container with `docker ps`.

## Skyvern Features

### Skyvern Tasks

Tasks are the core building blocks in Skyvern. Each task is a single request, instructing Skyvern to navigate and achieve a goal on a website.

Tasks require a `url`, `prompt`, and optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Screenshot">
</p>

### Skyvern Workflows

Workflows chain multiple tasks together.  For instance, downloading invoices, automating product purchases, or automating government forms.

Supported features:

1.  Navigation
2.  Action
3.  Data Extraction
4.  Loops
5.  File parsing
6.  Uploading files
7.  Sending emails
8.  Text Prompts
9.  Tasks
10. (Coming soon) Conditionals
11. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Invoice Download Workflow Example">
</p>

### Livestreaming

Watch Skyvern's browser viewport live for debugging and intervention.

### Form Filling

Natively fill out form inputs using the `navigation_goal`.

### Data Extraction

Extract structured data from websites.

Use a `data_extraction_schema` in JSONC format to define the desired output.

### File Downloading

Download files from websites. All downloaded files are uploaded to block storage.

### Authentication

Supports authentication methods to automate tasks behind logins.  Contact us at [founders@skyvern.com](mailto:founders@skyvern.com) or on [Discord](https://discord.gg/fG2XXEuQX3).

<p align="center">
  <img src="fern/images/secure_password_task_example.png" alt="Secure Password Task Example">
</p>

#### 🔐 2FA Support (TOTP)

Supports various 2FA methods:

1.  QR-based (Google Authenticator, Authy)
2.  Email-based
3.  SMS-based

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Supports the Model Context Protocol (MCP).

See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Integrate with Zapier, Make.com, and N8N to connect Skyvern workflows with other apps.

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-world Examples of Skyvern

See how Skyvern automates real-world workflows.  Submit your examples via PRs!

### Invoice Downloading on Many Websites

[Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo">
</p>

### Automate Job Applications

[💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Demo">
</p>

### Automate Materials Procurement

[💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center">
  <img src="fern/images/finditparts_recording_crop.gif" alt="Materials Procurement Demo">
</p>

### Navigate Government Websites

[💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="Government Website Demo">
</p>

### Filling Contact Us Forms

[💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Us Form Demo">
</p>

### Retrieving Insurance Quotes

[💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Demo">
</p>

[💡 See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Demo">
</p>

## Contributor Setup

For a complete local environment CLI installation:

```bash
pip install -e .
```

Set up the development environment using pre-commit:

```bash
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` to start using the UI.

    *Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Detailed documentation is available on our [📕 docs page](https://docs.skyvern.com). For any questions or issues, please open an issue or contact us at [founders@skyvern.com](mailto:founders@skyvern.com) or on [Discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider     | Supported Models                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| OpenAI       | gpt4-turbo, gpt-4o, gpt-4o-mini                                                                                              |
| Anthropic    | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                                          |
| Azure OpenAI | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                                      |
| AWS Bedrock  | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                                                                |
| Gemini       | Gemini 2.5 Pro and flash, Gemini 2.0                                                                                          |
| Ollama       | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)                                                   |
| OpenRouter   | Access models through [OpenRouter](https://openrouter.ai)                                                                     |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

##### OpenAI

| Variable              | Description                    | Type      | Sample Value       |
| --------------------- | ------------------------------ | --------- | ------------------ |
| `ENABLE_OPENAI`       | Register OpenAI models         | Boolean   | `true`, `false`   |
| `OPENAI_API_KEY`      | OpenAI API Key                 | String    | `sk-1234567890`    |
| `OPENAI_API_BASE`     | OpenAI API Base (Optional)     | String    | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | OpenAI Organization ID (Opt) | String    | `your-org-id`      |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

##### Anthropic

| Variable             | Description                  | Type      | Sample Value       |
| -------------------- | ---------------------------- | --------- | ------------------ |
| `ENABLE_ANTHROPIC`    | Register Anthropic models    | Boolean   | `true`, `false`   |
| `ANTHROPIC_API_KEY`   | Anthropic API key            | String    | `sk-1234567890`    |

Recommended `LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

##### Azure OpenAI

| Variable            | Description                       | Type      | Sample Value                     |
| ------------------- | --------------------------------- | --------- | -------------------------------- |
| `ENABLE_AZURE`      | Register Azure OpenAI models      | Boolean   | `true`, `false`                 |
| `AZURE_API_KEY`     | Azure deployment API key          | String    | `sk-1234567890`                  |
| `AZURE_DEPLOYMENT`  | Azure OpenAI Deployment Name      | String    | `skyvern-deployment`             |
| `AZURE_API_BASE`    | Azure deployment API base URL     | String    | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version                 | String    | `2024-02-01`                     |

Recommended `LLM_KEY`: `AZURE_OPENAI`

##### AWS Bedrock

| Variable                 | Description                                                                    | Type      | Sample Value       |
| ------------------------ | ------------------------------------------------------------------------------ | --------- | ------------------ |
| `ENABLE_BEDROCK`           | Register AWS Bedrock models. Requires proper AWS configurations.               | Boolean   | `true`, `false`   |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

##### Gemini

| Variable          | Description         | Type      | Sample Value                |
| ----------------- | ------------------- | --------- | --------------------------- |
| `ENABLE_GEMINI`    | Register Gemini models| Boolean   | `true`, `false`            |
| `GEMINI_API_KEY`   | Gemini API Key      | String    | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

##### Ollama

| Variable           | Description                 | Type      | Sample Value                |
| ------------------ | --------------------------- | --------- | --------------------------- |
| `ENABLE_OLLAMA`    | Register local models       | Boolean   | `true`, `false`            |
| `OLLAMA_SERVER_URL` | Ollama server URL           | String    | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL`     | Ollama model name to load | String    | `qwen2.5:7b-instruct`       |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

##### OpenRouter

| Variable             | Description                     | Type      | Sample Value                |
| -------------------- | ------------------------------- | --------- | --------------------------- |
| `ENABLE_OPENROUTER`    | Register OpenRouter models      | Boolean   | `true`, `false`            |
| `OPENROUTER_API_KEY`   | OpenRouter API key             | String    | `sk-1234567890`            |
| `OPENROUTER_MODEL`     | OpenRouter model name          | String    | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE`  | OpenRouter API base URL        | String    | `https://api.openrouter.ai/v1`  |

Recommended `LLM_KEY`: `OPENROUTER`

##### OpenAI-Compatible

| Variable                    | Description                                  | Type      | Sample Value                |
| --------------------------- | -------------------------------------------- | --------- | --------------------------- |
| `ENABLE_OPENAI_COMPATIBLE`  | Register a custom OpenAI-compatible endpoint | Boolean   | `true`, `false`            |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint | String    | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc.|
| `OPENAI_COMPATIBLE_API_KEY` | API key for OpenAI-compatible endpoint        | String    | `sk-1234567890`            |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint       | String    | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc.|
| `OPENAI_COMPATIBLE_API_VERSION` | API version for OpenAI-compatible endpoint, optional       | String    | `2023-05-15`|
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Maximum tokens for completion, optional      | Integer    | `4096`, `8192`, etc.|
| `OPENAI_COMPATIBLE_TEMPERATURE` | Temperature setting, optional               | Float      | `0.0`, `0.5`, `0.7`, etc.|
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Whether model supports vision, optional    | Boolean   | `true`, `false`|

Supported LLM Key: `OPENAI_COMPATIBLE`

##### General LLM Configuration

| Variable           | Description                                       | Type      | Sample Value          |
| ------------------ | ------------------------------------------------- | --------- | --------------------- |
| `LLM_KEY`          | The name of the model you want to use             | String    | See supported LLM keys above |
| `SECONDARY_LLM_KEY`| The name of the model for mini agents skyvern runs with      | String    | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM         | Integer    | `128000`              |

## Feature Roadmap

Our planned roadmap includes:

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

Contributions are welcome! Open PRs or issues, or reach out at [founders@skyvern.com](mailto:founders@skyvern.com) or on [Discord](https://discord.gg/fG2XXEuQX3).

See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for getting started!

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics by default. Set the `SKYVERN_TELEMETRY` environment variable to `false` to opt-out.

## License

Skyvern's core codebase is licensed under the [AGPL-3.0 License](LICENSE). Note: The anti-bot measures are available only in the managed cloud.

Contact us at [support@skyvern.com](mailto:support@skyvern.com) for licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)
```
Key improvements:

*   **SEO Optimization:**  Used more relevant keywords like "AI browser automation," "LLM automation," "web automation," and "RPA" throughout the README.
*   **Clear Headings and Structure:**  Improved heading hierarchy for better readability and organization.
*   **Concise Summary:**  Presented a clear and impactful one-sentence hook at the beginning.
*   **Bulleted Key Features:**  Used bullet points to highlight the main features, making them easy to scan.
*   **Actionable Quickstart:**  Made the quickstart section more user-friendly.
*   **Emphasis on Benefits:**  Highlighted advantages of using Skyvern over traditional automation methods.
*   **Consistent Formatting:** Applied consistent formatting across the document (e.g., bolding, code blocks).
*   **Clearer Language:**  Reworded some sentences for improved clarity and conciseness.
*   **Comprehensive LLM Support Section**: Improved the section on supported LLMs with better tables.
*   **Link back to the original repo:** Included the original repo link.
*   **Updated Demontration section**: Added a demo link.
*   **Added Code Sage reference**: Added a reference to code sage.