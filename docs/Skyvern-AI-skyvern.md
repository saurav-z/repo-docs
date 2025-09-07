<div align="center">
  <a href="https://www.skyvern.com">
    <img src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo" width="200">
  </a>
  <h1>Skyvern: Automate Browser Workflows with AI</h1>
  <p><strong>Harness the power of Large Language Models (LLMs) and Computer Vision to automate complex browser-based tasks.</strong></p>
  <p>
    <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"></a>
    <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"></a>
    <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"></a>
    <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"></a>
    <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"></a>
    <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"></a>
    <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"></a>
  </p>
</div>

[Skyvern](https://github.com/Skyvern-AI/skyvern) empowers you to automate repetitive browser tasks with ease, using the latest advancements in AI.  Replace brittle automation solutions and unlock efficiency with Skyvern.

## Key Features:

*   🌐 **Automated Web Navigation:** Navigate complex websites and interactions using LLMs and computer vision, no custom code needed.
*   ⚙️ **Workflow Automation:** Create and chain multiple tasks together for complex automation, supporting loops, data extraction, and more.
*   👁️ **Visual Understanding:** Operates effectively on websites it has never seen before by mapping visual elements and actions.
*   🛡️ **Resilient Automation:** Resistant to website layout changes; no reliance on fragile selectors like XPaths.
*   📊 **Data Extraction & Form Filling:** Extract structured data and fill forms seamlessly.
*   🔑 **Authentication Support:** Supports multiple authentication methods including 2FA (TOTP), for automating tasks behind logins, and password manager integration.
*   💻 **Easy Integration:** Integrates with tools like Zapier, Make.com, and N8N, and supports the Model Context Protocol (MCP).
*   ☁️ **Skyvern Cloud:** Try out the managed cloud version of Skyvern at [app.skyvern.com](https://app.skyvern.com)

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

Access the UI at [http://localhost:8080](http://localhost:8080) to run and monitor tasks.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

For more comprehensive instructions, please see the [Skyvern Documentation](https://docs.skyvern.com).

## How It Works

Skyvern leverages a swarm of intelligent agents, inspired by projects like BabyAGI and AutoGPT, and enhanced with browser automation via [Playwright](https://playwright.dev/). This architecture enables Skyvern to understand and interact with websites, regardless of their specific structure.  Skyvern also has the ability to reason through complex interactions, such as extracting data or making intelligent decisions on form filling, etc.
For a technical deep dive, see our report [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

<p align="center">
  <img src="fern/images/skyvern_2_0_system_diagram.png" alt="Skyvern Architecture Diagram">
</p>

## Demo

https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f

## Performance & Evaluation

Skyvern achieves industry-leading results on the [WebBench benchmark](webbench.ai), demonstrating superior performance in both read and write tasks.

<p align="center">
  <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance">
</p>

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Task Performance">
</p>

## Advanced Usage

### Control Your Own Browser (Chrome)

*   Just with Python Code:

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

*   With Skyvern Service (via .env file configuration):

```bash
CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BROWSER_TYPE=cdp-connect
```

### Run Skyvern with Any Remote Browser

```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get Consistent Output Schema from Your Run

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

### Helpful Commands to Debug Issues

```bash
skyvern run server     # Launch the Skyvern Server Separately
skyvern run ui         # Launch the Skyvern UI
skyvern status         # Check the Skyvern service status
skyvern stop all       # Stop the Skyvern service
skyvern stop ui        # Stop the Skyvern UI
skyvern stop server    # Stop the Skyvern Server Separately
```

## Docker Compose Setup

Follow these steps to set up Skyvern using Docker Compose.
[... Docker Instructions Here...]

## Skyvern Features

### Skyvern Tasks

Tasks are the core building blocks in Skyvern. Each task is a single request to Skyvern.

<p align="center">
  <img src="fern/images/skyvern_2_0_screenshot.png" alt="Skyvern Task Interface">
</p>

### Skyvern Workflows

Workflows allow you to chain tasks together to form cohesive units of work.

<p align="center">
  <img src="fern/images/invoice_downloading_workflow_example.png" alt="Workflow Example">
</p>

### Live Streaming

Skyvern streams the browser viewport to allow you to watch Skyvern take action on the web.

### Form Filling

Skyvern can fill out form inputs on websites.

### Data Extraction

Skyvern extracts data from websites.

### File Downloading

Skyvern downloads files from websites and uploads them to block storage (if configured).

### Authentication

Skyvern supports different authentication methods.

<p align="center">
  <img src="fern/images/secure_password_task_example.png" alt="Authentication Example">
</p>

## Real-World Examples

Discover how Skyvern is being used in production:

*   Invoice Downloading
*   Job Application Automation
*   Materials Procurement
*   Government Website Navigation
*   Contact Form Filling
*   Insurance Quote Retrieval

[...Insert Image and Link for each Example here...]

## Documentation

Find detailed information in our [Documentation](https://docs.skyvern.com). For questions or issues, contact us via [email](mailto:founders@skyvern.com) or [Discord](https://discord.gg/fG2XXEuQX3).

## Supported LLMs

[...Insert Table of Supported LLMs and Configuration Options Here...]

## Feature Roadmap

[...List Feature Roadmap Here...]

## Contributing

We welcome your contributions! Refer to the [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get involved.

## Telemetry

Skyvern collects basic usage statistics.  To opt out, set `SKYVERN_TELEMETRY=false` in your environment.

## License

Skyvern's core logic is licensed under the [AGPL-3.0 License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)