<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
    </picture>
  </a>
  <br>
  <p align="center"><strong>Automate Browser Workflows with the Power of LLMs and Computer Vision</strong></p>
  <p align="center">
    <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"></a>
    <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"></a>
    <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"></a>
    <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"></a>
    <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"></a>
    <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"></a>
    <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"></a>
  </p>
</h1>

[Skyvern](https://www.skyvern.com) is a powerful open-source tool that automates browser-based workflows using Large Language Models (LLMs) and computer vision, allowing you to automate complex tasks on any website without needing custom code.  Get started automating your browser tasks today!

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo">
</p>

## Key Features

*   **Automated Browser Automation**: Automate manual browser tasks with ease.
*   **LLM-Powered Interactions**: Interact with websites using the intelligence of LLMs and computer vision, adapting to website changes.
*   **No Code Required**: No need to write custom scripts for each website.
*   **Workflow Automation**: Chain multiple tasks together to create complex automated workflows.
*   **Data Extraction**: Extract structured data directly from websites.
*   **Form Filling**:  Seamlessly fill out forms on any website.
*   **File Downloading**: Download files automatically and upload them to block storage.
*   **Authentication Support**: Supports various authentication methods including 2FA.
*   **Model Context Protocol (MCP) Support**: Utilize any LLM that supports MCP.
*   **Integrations**: Ready-to-use integrations with Zapier, Make.com, and N8N.

Ready to see Skyvern in action? Check out the [real-world examples](#real-world-examples-of-skyvern) below!

## Quickstart

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern

```bash
skyvern quickstart
```

### 3. Run a Task

#### UI (Recommended)

Start the Skyvern service and access the UI:

```bash
skyvern run all
```

Then, go to http://localhost:8080 and create/run a task using the UI.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern will launch a browser to execute the task. You can view task history at http://localhost:8080/history.

You can also specify the environment where Skyvern runs:

```python
from skyvern import Skyvern

# Run on Skyvern Cloud (requires API key)
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Run on local Skyvern service
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## How Skyvern Works

Skyvern is inspired by autonomous agent designs like [BabyAGI](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT), but with the added ability to interact with websites using browser automation. Skyvern's key components are:

*   **Agents**: A swarm of agents that comprehends a website, plans actions, and executes them.
*   **Browser Automation**: Leveraging libraries like [Playwright](https://playwright.dev/) to interact with websites.
*   **LLM-Powered Reasoning**: Skyvern uses LLMs to reason through complex scenarios, such as handling variations in form questions or product descriptions.

Learn more about Skyvern's architecture in this [detailed technical report](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

Check out Skyvern in action:

https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern demonstrates state-of-the-art performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.  You can find the [technical report and evaluation here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/).

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance">
</p>

### Performance on WRITE Tasks

Skyvern excels in WRITE tasks, such as filling out forms, which is a key focus for RPA (Robotic Process Automation) use-cases.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Performance">
</p>

## Advanced Usage

### Control Your Own Chrome Browser

Use your local Chrome instance for tasks:

```python
from skyvern import Skyvern

browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # Example path for macOS
skyvern = Skyvern(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY",
    browser_path=browser_path,
)
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
```

### Run with a Remote Browser

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
```

### Consistent Output Schema

Define a data extraction schema for structured results:

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
    data_extraction_schema={...}
)
```

### Debugging Commands

```bash
skyvern run server
skyvern run ui
skyvern status
skyvern stop all
skyvern stop ui
skyvern stop server
```

## Docker Compose Setup

1.  Install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Clone the repository and navigate to the root directory.
3.  Run `skyvern init llm` to generate a `.env` file (configure your LLM API keys).
4.  Fill in your LLM provider key in the `docker-compose.yml` file.
5.  Run `docker compose up -d`.
6.  Access the UI at http://localhost:8080.

> **Important:**  If switching from the CLI-managed Postgres, remove the original container: `docker rm -f postgresql-container`.

## Skyvern Features (Detailed)

### Skyvern Tasks

Tasks are the basic instructions for Skyvern.  You define a `url`, `prompt`, and optional `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Screenshot">
</p>

### Skyvern Workflows

Workflows are chains of tasks.

Example: Download invoices newer than January 1st.

Supported workflow features:

1.  Navigation
2.  Action
3.  Data Extraction
4.  Loops
5.  File parsing
6.  Uploading files to block storage
7.  Sending emails
8.  Text Prompts
9.  Tasks
10. (Coming soon) Conditionals
11. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Invoice Downloading Workflow Example">
</p>

### Livestreaming

Livestream the browser viewport to see Skyvern in action.

### Form Filling

Skyvern natively fills out form inputs.

### Data Extraction

Extract data from websites with structured output.

### File Downloading

Download and automatically upload files to block storage.

### Authentication

Supports various authentication methods.

### 🔐 2FA Support (TOTP)

Supports 2FA methods.
Learn more [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

Supports Bitwarden (and soon others).

### Model Context Protocol (MCP)

Supports any LLM with MCP.
See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)

### Zapier / Make.com / N8N Integration

Connect to other apps.
*   [Zapier](https://docs.skyvern.com/integrations/zapier)
*   [Make.com](https://docs.skyvern.com/integrations/make.com)
*   [N8N](https://docs.skyvern.com/integrations/n8n)

## Real-world examples of Skyvern

See Skyvern in action:

### Invoice Downloading

[Book a demo](https://meetings.hubspot.com/skyvern/demo)

<p align="center">
  <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo">
</p>

### Automate Job Applications

[See it in action](https://app.skyvern.com/tasks/create/job_application)

<p align="center">
  <img src="fern/images/job_application_demo.gif" alt="Job Application Demo">
</p>

### Automate Materials Procurement

[See it in action](https://app.skyvern.com/tasks/create/finditparts)

<p align="center>
  <img src="fern/images/finditparts_recording_crop.gif" alt="Finditparts Recording">
</p>

### Navigating Government Websites

[See it in action](https://app.skyvern.com/tasks/create/california_edd)

<p align="center">
  <img src="fern/images/edd_services.gif" alt="California EDD Services">
</p>

### Filling Out Contact Forms

[See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

<p align="center">
  <img src="fern/images/contact_forms.gif" alt="Contact Forms Demo">
</p>

### Retrieving Insurance Quotes

[See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

<p align="center">
  <img src="fern/images/bci_seguros_recording.gif" alt="BCI Seguros Demo">
</p>

[See it in action](https://app.skyvern.com/tasks/create/geico)

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Demo">
</p>

## Contributor Setup

```bash
pip install -e .
skyvern quickstart contributors
```

1.  Navigate to `http://localhost:8080` in your browser.

*Skyvern CLI supports Windows, WSL, macOS, and Linux.*

## Documentation

Extensive documentation available on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

| Provider        | Supported Models                                                                   |
| --------------- | ---------------------------------------------------------------------------------- |
| OpenAI          | gpt4-turbo, gpt-4o, gpt-4o-mini                                                  |
| Anthropic       | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                             |
| Azure OpenAI    | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)           |
| AWS Bedrock     | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)                     |
| Gemini          | Gemini 2.5 Pro and flash, Gemini 2.0                                             |
| Ollama          | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)         |
| OpenRouter      | Access models through [OpenRouter](https://openrouter.ai)                         |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

### Environment Variables (LLM Configuration)

See the original README for complete details.

## Feature Roadmap

Planned features:
*   [ ] **Improved Debug mode**
*   [ ] **Chrome Extension**
*   [ ] **Skyvern Action Recorder**
*   [ ] **Interactable Livestream**
*   [ ] **Integrate LLM Observability tools**
*   [x] **Langchain Integration**

## Contributing

Contributions are welcome!  See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for getting started.

For high-level assistance with the Skyvern project, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

Skyvern collects basic usage statistics. To opt-out, set `SKYVERN_TELEMETRY` to `false`.

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE).  See the managed cloud for exception details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)