<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  Skyvern: Automate Your Browser Workflows with AI 🐉
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

<p align="center">
  Effortlessly automate browser-based tasks with Skyvern, utilizing Large Language Models (LLMs) and computer vision to navigate and interact with websites as a human would. <a href="https://github.com/Skyvern-AI/skyvern">Learn More</a>
</p>

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo"/>
</p>

## Key Features

*   **Intelligent Automation:** Leverage LLMs and computer vision to interact with websites, adapting to layout changes.
*   **Workflow Automation:** Chain multiple tasks together for complex automated processes.
*   **Data Extraction:** Easily extract structured data from web pages.
*   **Form Filling:** Automate form submissions on any website.
*   **2FA Support:** Securely automate tasks with 2FA (TOTP, email, SMS)
*   **Livestreaming:** See Skyvern's actions in real-time with browser viewport streaming.
*   **Integration:** Supports Zapier, Make.com, and N8N.

## Quickstart

### Installation

```bash
pip install skyvern
```

### Run the UI

```bash
skyvern run all
```

Visit [http://localhost:8080](http://localhost:8080) in your browser to run tasks via the UI.

## How Skyvern Works

Skyvern uses a swarm of AI agents that are powered by a multimodal LLM and a Playwright browser to comprehend a website, and plan and execute its actions.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_2_0_system_diagram.png" />
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern System Diagram"/>
</picture>

Skyvern offers:

*   Website Agnosticism
*   Resistance to Layout Changes
*   Scalable Workflow Automation
*   LLM-Powered Reasoning

Find the detailed technical report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Performance & Evaluation

Skyvern has SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy. The technical report + evaluation can be found [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

<p align="center">
  <img src="fern/images/performance/webbench_overall.png"/>
</p>

## Performance on WRITE tasks (eg filling out forms, logging in, downloading files, etc)

Skyvern is the best performing agent on WRITE tasks (eg filling out forms, logging in, downloading files, etc), which is primarily used for RPA (Robotic Process Automation) adjacent tasks.

<p align="center">
  <img src="fern/images/performance/webbench_write.png"/>
</p>

## Advanced Usage

### Control Your Own Browser

1.  **With Python Code:**
    ```python
    from skyvern import Skyvern

    browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # Example path for Mac
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
    Add the following to your `.env` file:
    ```bash
    CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    BROWSER_TYPE=cdp-connect
    ```
    Restart the Skyvern service `skyvern run all` and run the task via UI or code.

### Run with Any Remote Browser

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

Follow the steps to set up Skyvern with Docker Compose.

## Skyvern Features

### Skyvern Tasks

Each task is a request to Skyvern, to automate a goal.

Tasks use: `url`, `prompt`, `data schema`, `error codes`.

### Skyvern Workflows

Chaining tasks for more complex automation (e.g., download invoices).

Features include:

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

Watch the browser viewport live.

### Form Filling

Automated form completion.

### Data Extraction

Extract structured data using a defined schema.

### File Downloading

Download files, automatically upload them to block storage, and access them via the UI.

### Authentication

Support for various authentication methods.

*   **2FA Support (TOTP)**
    *   QR-based 2FA
    *   Email-based 2FA
    *   SMS-based 2FA

*   **Password Manager Integrations**
    *   Bitwarden

    [Read more](https://docs.skyvern.com/credentials/totp).

## Real-World Examples

*   Invoice Downloading
*   Job Application Automation
*   Materials Procurement Automation
*   Government Website Navigation
*   Contact Form Filling
*   Insurance Quote Retrieval

[See Skyvern in action](https://app.skyvern.com/tasks/create/job_application).

## Documentation

Extensive documentation is available on our [docs page](https://docs.skyvern.com).

## Supported LLMs

See the table detailing supported LLMs and environment variables.

## Feature Roadmap

The planned roadmap for the coming months.

## Contributing

Contributions are welcome!  See the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Telemetry

Skyvern collects basic usage statistics. To opt-out, set `SKYVERN_TELEMETRY=false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).