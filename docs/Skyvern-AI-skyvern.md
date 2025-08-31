<h1 align="center">
 <a href="https://www.skyvern.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png"/>
  </picture>
 </a>
 <br />
</h1>

<p align="center">
  <strong>Automate any browser-based workflow with the power of LLMs and Computer Vision.</strong>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

[Skyvern](https://www.skyvern.com) empowers you to automate complex browser-based tasks using the intelligence of Large Language Models (LLMs) and the visual understanding of Computer Vision.  Say goodbye to brittle, code-dependent automation and hello to a more robust and adaptable approach.

## Key Features

*   ✅ **Intelligent Automation:**  Skyvern uses LLMs to *understand* websites and *execute* actions, adapting to changes without requiring code modifications.
*   ✅ **Versatile:** Automate workflows across a wide range of websites, even those you've never encountered before.
*   ✅ **Robust:** Resistant to website layout changes; Skyvern learns and interacts visually.
*   ✅ **Workflow Capabilities:** Chain multiple tasks together with support for navigation, actions, data extraction, loops, file handling, and more.
*   ✅ **Advanced Functionality:** Includes form filling, data extraction, file downloading, authentication (including 2FA), and integration with tools like Zapier and Make.com.
*   ✅ **Livestreaming and Debugging:** Monitor Skyvern's actions in real time with browser viewport livestreaming.
*   ✅ **Comprehensive LLM Support:** Compatible with a wide variety of LLMs (OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Gemini, Ollama, OpenRouter, and custom OpenAI-compatible endpoints) - [See Supported LLMs](#supported-llms).
*   ✅ **Easy Integration:** Supports Zapier, Make.com, and N8N for connecting your workflows to other apps.

Want to see Skyvern in action?  Check out our [#real-world-examples-of-skyvern](#real-world-examples-of-skyvern)

## Quickstart

### 1. Installation

```bash
pip install skyvern
```

### 2. Run Skyvern

Choose your preferred method:

#### UI (Recommended)

Start the Skyvern service and UI:

```bash
skyvern run all
```

Then, access the UI at http://localhost:8080 to run tasks.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern launches and closes a browser as needed. View task history at http://localhost:8080/history.

You can also specify different target environments:

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

Inspired by task-driven autonomous agents like BabyAGI and AutoGPT, Skyvern leverages a swarm of agents and browser automation to navigate and interact with websites.

[<img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram" width="800"/>](fern/images/skyvern_2_0_system_diagram.png)

This design offers several advantages:

*   **Zero-Code Automation:** Operate on websites without requiring custom code for each site.
*   **Layout Change Resilience:**  Skyvern's visual understanding makes it resistant to website layout updates.
*   **Cross-Site Application:**  Apply the same workflow across many websites.
*   **LLM-Powered Reasoning:**  LLMs handle complex scenarios such as inferring driver eligibility questions on insurance forms.

A detailed technical report is available [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<!-- Redo demo -->
https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. The technical report and evaluation can be found [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

<p align="center">
  <img src="fern/images/performance/webbench_overall.png"/>
</p>

### Performance on WRITE Tasks

Skyvern excels in WRITE tasks (e.g., form filling, logins, downloads), essential for RPA.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

## Advanced Usage

### Control your own browser (Chrome)
> ⚠️ WARNING: Since [Chrome 136](https://developer.chrome.com/blog/remote-debugging-port), Chrome refuses any CDP connect to the browser using the default user_data_dir. In order to use your browser data, Skyvern copies your default user_data_dir to `./tmp/user_data_dir` the first time connecting to your local browser. ⚠️

1. Just With Python Code
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

2. With Skyvern Service

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

## Docker Compose Setup

(Instructions for Docker Compose setup remain the same)

## Skyvern Features

### Skyvern Tasks

Tasks are the core units within Skyvern, representing a single instruction to automate a web workflow. Tasks require a `url`, `prompt`, and can optionally include a `data schema` and `error codes`.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows

Workflows streamline complex tasks by chaining multiple Skyvern tasks.

Example: Downloading invoices.  A workflow could navigate to an invoices page, filter for a date range, extract invoice details, and download each invoice.

Supported features include:

1.  Navigation
2.  Action
3.  Data Extraction
4.  Loops
5.  File Parsing
6.  File Uploading to Block Storage
7.  Sending Emails
8.  Text Prompts
9.  Tasks (general)
10. (Coming soon) Conditionals
11. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png"/>
</p>

### Livestreaming

View the browser's viewport in real time for debugging and monitoring.

### Form Filling

Skyvern natively handles form input. The `navigation_goal` helps Skyvern comprehend and fill forms.

### Data Extraction

Extract structured data from websites. You can define a  `data_extraction_schema` in your prompt for JSON-formatted output.

### File Downloading

Skyvern can download files from websites. Downloaded files are automatically uploaded to block storage (if configured) and accessible in the UI.

### Authentication

Supports various authentication methods. Contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3) for details.

<p align="center">
  <img src="fern/images/secure_password_task_example.png"/>
</p>

### 🔐 2FA Support (TOTP)

Skyvern automates workflows requiring 2FA:

1.  QR-based (Google Authenticator, Authy, etc.)
2.  Email-based
3.  SMS-based

Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).

### Password Manager Integrations

*   [x] Bitwarden
*   [ ] 1Password
*   [ ] LastPass

## Real-world Examples of Skyvern

Discover how Skyvern is being used in the real world.  We encourage you to add your own examples via pull requests!

*   **Invoice Downloading:** Automate invoice downloads from various websites.  [Book a demo](https://meetings.hubspot.com/skyvern/demo)

    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   **Job Application Automation:** Simplify the job application process.  [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)

    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>
*   **Materials Procurement:** Automate material procurement for manufacturing.  [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)

    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>
*   **Government Website Automation:** Register accounts and fill out forms on government websites.  [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)

    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>
*   **Contact Form Automation:**  Automate filling out "contact us" forms.  [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)

    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>
*   **Insurance Quote Retrieval:**  Retrieve insurance quotes in any language. [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)

    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)

    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Contributor Setup

(Instructions for Contributor setup remain the same)

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com). Contact us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3) if you have any questions.

## Supported LLMs

| Provider         | Supported Models                                               |
| ---------------- | ------------------------------------------------------------ |
| OpenAI           | gpt4-turbo, gpt-4o, gpt-4o-mini                              |
| Anthropic        | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)          |
| Azure OpenAI     | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o) |
| AWS Bedrock      | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)          |
| Gemini           | Gemini 2.5 Pro and flash, Gemini 2.0                        |
| Ollama           | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama) |
| OpenRouter       | Access models through [OpenRouter](https://openrouter.ai)  |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)) |

#### Environment Variables

(Environment Variables table remains the same)

## Feature Roadmap

(Feature roadmap remains the same)

## Contributing

We welcome contributions!  Please see our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

If you want to chat with the skyvern repository to get a high level overview of how it is structured, how to build off it, and how to resolve usage questions, check out [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme).

## Telemetry

(Telemetry information remains the same)

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE).

If you have any questions or concerns around licensing, please [contact us](mailto:support@skyvern.com) and we would be happy to help.

## Star History

(Star History Chart remains the same)