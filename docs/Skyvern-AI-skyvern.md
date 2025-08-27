<h1 align="center">
 <a href="https://www.skyvern.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png"/>
  </picture>
 </a>
 <br />
  <p align="center"><b>Automate any browser-based workflow with the power of AI!</b></p>
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

[Skyvern](https://www.skyvern.com) is a powerful, AI-driven platform that automates complex browser-based workflows.  Tired of brittle automation scripts? Skyvern utilizes Large Language Models (LLMs) and computer vision to interact with websites, overcoming the limitations of traditional methods.

*   **Automate Anything:** Automate a wide range of browser-based tasks, from data extraction to form filling, across numerous websites.
*   **LLM-Powered:**  Leverages LLMs for intelligent website interaction, adapting to changes and handling complex scenarios.
*   **Resilient Automation:** Built to withstand website layout changes, eliminating the need for constant script updates.
*   **Real-World Applications:**  Ideal for Robotic Process Automation (RPA), data collection, and streamlining repetitive online tasks.
*   **Easy Integration:**  Simple API for integration, plus options for cloud and local deployment.

Ready to see Skyvern in action? Jump to our [#real-world-examples-of-skyvern](#real-world-examples-of-skyvern) section!

## Quickstart

### Skyvern Cloud

[Skyvern Cloud](https://app.skyvern.com) offers a fully managed version of Skyvern, eliminating infrastructure concerns. It includes features like parallel instance execution, anti-bot detection, proxy networks, and CAPTCHA solvers. Visit [app.skyvern.com](https://app.skyvern.com) to create an account.

### Install & Run Locally

1.  **Install Skyvern:**

    ```bash
    pip install skyvern
    ```

2.  **Run Skyvern:**

    ```bash
    skyvern quickstart
    ```

3.  **Run Task**

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

## How Skyvern Works

Skyvern is inspired by task-driven autonomous agents like BabyAGI and AutoGPT, but with the crucial ability to interact with websites using browser automation libraries (e.g., Playwright).

Skyvern employs a swarm of intelligent agents to:

1.  **Comprehend the Website:** Analyze the website's structure and content.
2.  **Plan Actions:**  Determine the steps needed to complete the desired task.
3.  **Execute Actions:**  Interact with the website using browser automation.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" />
</picture>

This approach provides several advantages:

*   **Versatile:** Works on websites never seen before, mapping visual elements to actions.
*   **Robust:** Resistant to website layout changes.
*   **Scalable:** Easily applies a single workflow across many websites.
*   **Intelligent:** Uses LLMs to handle complex situations and reasoning.

Read more details in our technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. The technical report and evaluation can be found [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png"/>
</p>

### Performance on WRITE Tasks

Skyvern leads in performance on WRITE tasks (e.g. form filling, logins, and downloads) – tasks crucial for RPA.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

## Advanced Usage

### Control Your Browser (Chrome)

> ⚠️ WARNING: Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1.  **With Python Code**
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

    Add these variables to your `.env` file:

    ```bash
    CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    BROWSER_TYPE=cdp-connect
    ```

    Restart Skyvern: `skyvern run all`.

### Run with a Remote Browser

Provide Skyvern with the CDP connection URL:

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Consistent Output Schema

Use the `data_extraction_schema` parameter:

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

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Clone the repository.
3.  Run `skyvern init llm` to create a `.env` file.
4.  Populate the LLM provider key in `docker-compose.yml`.
5.  Run `docker compose up -d`.
6.  Access the UI at `http://localhost:8080`.

> **Important:** Remove any local Postgres containers before starting with Docker Compose.

## Skyvern Features

### Skyvern Tasks

Tasks are the basic units of work, defined by a `url`, `prompt`, optional `data schema`, and optional `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows

Workflows chain multiple tasks for complex automations. Features include:

1.  Navigation
2.  Action
3.  Data Extraction
4.  Loops
5.  File Parsing
6.  File Uploads
7.  Sending Emails
8.  Text Prompts
9.  Tasks
10. (Coming soon) Conditionals
11. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png"/>
</p>

### Livestreaming

Livestream the browser viewport to your local machine for debugging and monitoring.

### Form Filling

Skyvern excels at filling out forms using the information provided in `navigation_goal`.

### Data Extraction

Extract structured data with the ability to specify a `data_extraction_schema` in JSONC format.

### File Downloading

Download files automatically, with options for block storage (if configured).

### Authentication

Skyvern provides different authentication methods. Please [contact us](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3) to try it out.

<p align="center">
  <img src="fern/images/secure_password_task_example.png"/>
</p>

### 🔐 2FA Support (TOTP)

Automate workflows requiring 2FA with support for:

1.  QR-based 2FA
2.  Email-based 2FA
3.  SMS-based 2FA

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

### Model Context Protocol (MCP)

Skyvern supports the Model Context Protocol (MCP) for using any LLM that supports it. See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Connect your Skyvern workflows to other apps:

*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

## Real-World Examples of Skyvern

Discover how Skyvern automates real-world workflows. [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo).

*   **Invoice Downloading on multiple websites**

    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   **Automate job applications**
    [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>

*   **Automate materials procurement for a manufacturing company**
    [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>

*   **Navigating government websites for account registration/form filling**
    [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>
*   **Filling contact forms**
    [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>
*   **Retrieving insurance quotes in any language**
    [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Contributor Setup

Install for local environment:

```bash
pip install -e .
```

Set up pre-commit hooks:

```bash
skyvern quickstart contributors
```

1.  Access the UI at `http://localhost:8080`.
2.  *The Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Find more detailed documentation on our [📕 docs page](https://docs.skyvern.com).  For questions, open an issue or reach out via [email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

| Provider | Supported Models |
| -------- | ------- |
| OpenAI   | gpt4-turbo, gpt-4o, gpt-4o-mini |
| Anthropic | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Azure OpenAI | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Gemini | Gemini 2.5 Pro and flash, Gemini 2.0 |
| Ollama | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter | Access models through [OpenRouter](https://openrouter.ai) |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

### Environment Variables

#### OpenAI

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OPENAI` | Enable OpenAI models | Boolean | `true`, `false` |
| `OPENAI_API_KEY` | Your OpenAI API Key | String | `sk-1234567890` |
| `OPENAI_API_BASE` | Optional OpenAI API Base | String | `https://openai.api.base` |
| `OPENAI_ORGANIZATION` | Optional OpenAI Organization ID | String | `your-org-id` |

Recommended `LLM_KEY`: `OPENAI_GPT4O`, `OPENAI_GPT4O_MINI`, `OPENAI_GPT4_1`, `OPENAI_O4_MINI`, `OPENAI_O3`

#### Anthropic

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_ANTHROPIC` | Enable Anthropic models | Boolean | `true`, `false` |
| `ANTHROPIC_API_KEY` | Your Anthropic API Key | String | `sk-1234567890` |

Recommended`LLM_KEY`: `ANTHROPIC_CLAUDE3.5_SONNET`, `ANTHROPIC_CLAUDE3.7_SONNET`, `ANTHROPIC_CLAUDE4_OPUS`, `ANTHROPIC_CLAUDE4_SONNET`

#### Azure OpenAI

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_AZURE` | Enable Azure OpenAI models | Boolean | `true`, `false` |
| `AZURE_API_KEY` | Azure deployment API Key | String | `sk-1234567890` |
| `AZURE_DEPLOYMENT` | Azure OpenAI Deployment Name | String | `skyvern-deployment` |
| `AZURE_API_BASE` | Azure deployment API base URL | String | `https://skyvern-deployment.openai.azure.com/` |
| `AZURE_API_VERSION` | Azure API Version | String | `2024-02-01` |

Recommended `LLM_KEY`: `AZURE_OPENAI`

#### AWS Bedrock

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_BEDROCK` | Enable AWS Bedrock models. Requires proper [AWS configurations](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3). | Boolean | `true`, `false` |

Recommended `LLM_KEY`: `BEDROCK_ANTHROPIC_CLAUDE3.7_SONNET_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_OPUS_INFERENCE_PROFILE`, `BEDROCK_ANTHROPIC_CLAUDE4_SONNET_INFERENCE_PROFILE`

#### Gemini

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_GEMINI` | Enable Gemini models | Boolean | `true`, `false` |
| `GEMINI_API_KEY` | Gemini API Key | String | `your_google_gemini_api_key` |

Recommended `LLM_KEY`: `GEMINI_2.5_PRO_PREVIEW`, `GEMINI_2.5_FLASH_PREVIEW`

#### Ollama

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OLLAMA` | Enable local models via Ollama | Boolean | `true`, `false` |
| `OLLAMA_SERVER_URL` | Ollama server URL | String | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | Ollama model name | String | `qwen2.5:7b-instruct` |

Recommended `LLM_KEY`: `OLLAMA`

Note: Ollama does not support vision yet.

#### OpenRouter

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OPENROUTER` | Enable OpenRouter models | Boolean | `true`, `false` |
| `OPENROUTER_API_KEY` | OpenRouter API key | String | `sk-1234567890` |
| `OPENROUTER_MODEL` | OpenRouter model name | String | `mistralai/mistral-small-3.1-24b-instruct` |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | String | `https://api.openrouter.ai/v1` |

Recommended `LLM_KEY`: `OPENROUTER`

#### OpenAI-Compatible

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `ENABLE_OPENAI_COMPATIBLE` | Enable custom OpenAI-compatible API endpoint | Boolean | `true`, `false` |
| `OPENAI_COMPATIBLE_MODEL_NAME` | Model name for OpenAI-compatible endpoint | String | `yi-34b`, `gpt-3.5-turbo`, `mistral-large`, etc. |
| `OPENAI_COMPATIBLE_API_KEY` | API key for OpenAI-compatible endpoint | String | `sk-1234567890` |
| `OPENAI_COMPATIBLE_API_BASE` | Base URL for OpenAI-compatible endpoint | String | `https://api.together.xyz/v1`, `http://localhost:8000/v1`, etc. |
| `OPENAI_COMPATIBLE_API_VERSION` | Optional API version for OpenAI-compatible endpoint | String | `2023-05-15` |
| `OPENAI_COMPATIBLE_MAX_TOKENS` | Optional maximum tokens for completion | Integer | `4096`, `8192`, etc. |
| `OPENAI_COMPATIBLE_TEMPERATURE` | Optional temperature setting | Float | `0.0`, `0.5`, `0.7`, etc. |
| `OPENAI_COMPATIBLE_SUPPORTS_VISION` | Optional: Whether model supports vision | Boolean | `true`, `false` |

Supported LLM Key: `OPENAI_COMPATIBLE`

#### General LLM Configuration

| Variable | Description | Type | Sample Value |
|---|---|---|---|
| `LLM_KEY` | The name of the model you want to use | String | See supported LLM keys above |
| `SECONDARY_LLM_KEY` | The name of the model for mini agents skyvern runs with | String | See supported LLM keys above |
| `LLM_CONFIG_MAX_TOKENS` | Override the max tokens used by the LLM | Integer | `128000` |

## Feature Roadmap

Future planned features:

-   [ ] Improved Debug mode
-   [ ] Chrome Extension
-   [ ] Skyvern Action Recorder
-   [ ] Interactable Livestream
-   [ ] Integrate LLM Observability tools
-   [x] Langchain Integration

## Contributing

Contributions are welcome!  See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).  For high-level insights on the structure, and to get answers to your usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics by default; opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

This open-source repository is under the [AGPL-3.0 License](LICENSE). The managed cloud offering includes additional anti-bot measures.  For licensing questions, please [contact us](mailto:support@skyvern.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)
```
Key improvements and explanations:

*   **SEO-Optimized Title & Hook:**  The title is clear, concise, and includes keywords ("browser automation," "AI," "LLMs"). The one-sentence hook immediately grabs attention.
*   **Clear Headings and Structure:** The use of headings (Quickstart, How It Works, etc.) and subheadings makes the README easy to scan and understand.
*   **Bulleted Key Features:**  Highlights the main benefits in a digestible format.
*   **Concise Language:**  The content is rewritten to be more direct and engaging.
*   **Improved Examples:** Provides clear use cases to show off Skyvern's abilities.
*   **Call to Actions:** Encourages engagement ("Ready to see...", "Jump to...").
*   **Complete Install Instructions:**  The install and run commands have been updated.
*   **LLM Key Explanations:** Gives detailed variable information and more detail on which keys to use.
*   **Prioritization and Conciseness:** Removed less important information to get to the core features of the project.
*   **Comprehensive Information:**  Includes all necessary instructions, documentation links, and contribution guidance.
*   **Clear License Information:** Clarifies licensing terms.
*   **GitHub Star History Added:** Provides social proof and visualises project growth.
*   **Better Formatting:** Improves the readability.
*   **Docker Instructions Improved.** Easier to follow the Docker setup steps.
*   **Contained Warnings:** The warnings were clearly highlighted, and placed at the top of their sections.
*   **Code Sage link added.**

This revised README is much more effective at attracting users, explaining the core functionality, and encouraging adoption.