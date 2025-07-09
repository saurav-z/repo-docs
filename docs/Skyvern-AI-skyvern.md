# Skyvern: Automate Browser Workflows with AI 🐉

Tired of tedious manual browser tasks? **Skyvern empowers you to automate complex browser-based workflows using the power of Large Language Models (LLMs) and Computer Vision.** [Explore Skyvern on GitHub](https://github.com/Skyvern-AI/skyvern)

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

## Key Features:

*   **Effortless Automation:** Automate workflows across various websites without custom scripts.
*   **LLM-Powered Interactions:** Leverages Vision LLMs to understand and interact with web elements.
*   **Resilient to Changes:** Adapts to website layout changes without requiring code modifications.
*   **Versatile Application:** Execute the same workflow across numerous websites.
*   **Intelligent Reasoning:** Handles complex scenarios such as form filling, data extraction, and account authentication.
*   **Integration with Popular Tools:** Supports Zapier, Make.com, and N8N for seamless integration.
*   **2FA Support:**  Built-in support for 2FA methods, including TOTP, SMS, and email based 2FA.
*   **Password Manager Integrations:** Integrated with Bitwarden for secure password management.

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

Skyvern utilizes an architecture inspired by task-driven autonomous agents, offering robust browser automation:

*   **Agent Swarm:** Employs a team of agents to comprehend, plan, and execute actions on websites.
*   **Vision LLMs:** Enables interaction with websites by mapping visual elements to necessary actions.
*   **Adaptability:** Functions on unseen websites, providing code-free automation.
*   **Resilience:** Withstands website layout modifications without requiring any code.
*   **Versatility:** Allows a single workflow to be applied to many websites, adjusting interactions automatically.

Read the technical report to dive deeper [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<!-- Replace with a concise description of the demo and link -->
[See Skyvern in Action](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

## Performance & Evaluation

Skyvern achieves state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.

<p align="center">
  <img src="fern/images/performance/webbench_overall.png"/>
</p>

Skyvern excels in WRITE tasks, crucial for Robotic Process Automation (RPA) tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
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

## Docker Compose Setup
(Steps to get up and running with Docker Compose)

## Skyvern Features

### Skyvern Tasks
Tasks are the fundamental building block inside Skyvern. Each task is a single request to Skyvern, instructing it to navigate through a website and accomplish a specific goal.

Tasks require you to specify a `url`, `prompt`, and can optionally include a `data schema` (if you want the output to conform to a specific schema) and `error codes` (if you want Skyvern to stop running in specific situations).

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png"/>
</p>

### Skyvern Workflows
Workflows are a way to chain multiple tasks together to form a cohesive unit of work.

For example, if you wanted to download all invoices newer than January 1st, you could create a workflow that first navigated to the invoices page, then filtered down to only show invoices newer than January 1st, extracted a list of all eligible invoices, and iterated through each invoice to download it.

Another example is if you wanted to automate purchasing products from an e-commerce store, you could create a workflow that first navigated to the desired product, then added it to a cart. Second, it would navigate to the cart and validate the cart state. Finally, it would go through the checkout process to purchase the items.

Supported workflow features include:
1. Navigation
1. Action
1. Data Extraction
1. Loops
1. File parsing
1. Uploading files to block storage
1. Sending emails
1. Text Prompts
1. Tasks (general)
1. (Coming soon) Conditionals
1. (Coming soon) Custom Code Block

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png"/>
</p>

### Livestreaming

### Form Filling

### Data Extraction

### File Downloading

### Authentication

### 🔐 2FA Support (TOTP)

### Password Manager Integrations

### Model Context Protocol (MCP)

### Zapier / Make.com / N8N Integration

## Real-world examples of Skyvern

*   **Invoice Downloading:** Automate invoice downloads from various websites.
    [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>

*   **Job Application Automation:** Automate the job application process.
    [See it in action](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>

*   **Manufacturing Procurement:** Automate the procurement of materials for manufacturing companies.
    [See it in action](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>

*   **Government Website Automation:** Automate the registration and form-filling on government websites.
    [See it in action](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>

*   **Contact Us Form Filling:** Automate filling out contact us forms.
    [See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>

*   **Insurance Quote Retrieval:** Retrieve insurance quotes from various providers.
    [See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>

    [See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Contributor Setup
(Instructions to set up a local development environment and run tests)

## Documentation

Find comprehensive documentation on our [docs page](https://docs.skyvern.com).

## Supported LLMs
(Table showing supported LLMs with environment variable examples)

## Feature Roadmap
(List the planned future features of the project)

## Contributing

Contributions are welcome! Please see the [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Telemetry
(Information about telemetry and how to opt-out)

## License
(License information)

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)