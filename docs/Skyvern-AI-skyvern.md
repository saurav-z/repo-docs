<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br/>
  Skyvern: Automate Web Workflows with AI
</h1>

<p align="center">
  <b>Unlock the power of AI to automate complex browser-based tasks with Skyvern.</b>
  <br/>
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Docs"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

[Skyvern](https://github.com/Skyvern-AI/skyvern) leverages the power of Large Language Models (LLMs) and computer vision to automate intricate browser-based workflows, providing a robust solution for automating manual processes across a multitude of websites.

## Key Features

*   🚀 **Intelligent Automation:** Automates tasks on any website without the need for custom code, adapting to website changes.
*   🤖 **Vision-Powered Navigation:** Utilizes Vision LLMs to understand and interact with websites, enabling dynamic interactions.
*   🔗 **Workflow Orchestration:** Chains multiple tasks for complex automation, including data extraction, file handling, and more.
*   🌐 **Cross-Website Compatibility:** A single workflow can be applied across various websites due to the AI's reasoning capabilities.
*   ✅ **Robust Form Filling:** Easily handles form inputs on websites with intelligent data comprehension.
*   📤 **Data Extraction & File Downloading:** Extract structured data and download files directly from websites.
*   🛡️ **Authentication Support:** Includes support for various authentication methods.

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

Visit http://localhost:8080 to run a task via the UI.

#### Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern will launch a browser, complete the task, and close. View task history at http://localhost:8080/history.

You can also specify targets:
```python
from skyvern import Skyvern

# Run on Skyvern Cloud (replace with your API key)
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Local Skyvern service
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## [How Skyvern Works](https://github.com/Skyvern-AI/skyvern#how-it-works)

Skyvern uses a swarm of intelligent agents, inspired by concepts like BabyAGI and AutoGPT, enhancing them with browser interaction through libraries like Playwright. This results in adaptability and resistance to website changes.

## [Demo](https://github.com/user-attachments/assets/5cab4668-e8e2-4982-8551-aab05ff73a7f)

See Skyvern in action with a demo.

## [Performance & Evaluation](https://github.com/Skyvern-AI/skyvern#performance--evaluation)

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai).

## Advanced Usage

### [Control Your Own Browser (Chrome)](https://github.com/Skyvern-AI/skyvern#control-your-own-browser-chrome)
### [Run Skyvern with any remote browser](https://github.com/Skyvern-AI/skyvern#run-skyvern-with-any-remote-browser)
### [Get Consistent Output Schema](https://github.com/Skyvern-AI/skyvern#get-consistent-output-schema-from-your-run)
### [Helpful Debug Commands](https://github.com/Skyvern-AI/skyvern#helpful-commands-to-debug-issues)

## [Docker Compose Setup](https://github.com/Skyvern-AI/skyvern#docker-compose-setup)

Steps for setting up Skyvern using Docker Compose.

## [Skyvern Features](https://github.com/Skyvern-AI/skyvern#skyvern-features)

### [Skyvern Tasks](https://github.com/Skyvern-AI/skyvern#skyvern-tasks)
### [Skyvern Workflows](https://github.com/Skyvern-AI/skyvern#skyvern-workflows)
### [Livestreaming](https://github.com/Skyvern-AI/skyvern#livestreaming)
### [Form Filling](https://github.com/Skyvern-AI/skyvern#form-filling)
### [Data Extraction](https://github.com/Skyvern-AI/skyvern#data-extraction)
### [File Downloading](https://github.com/Skyvern-AI/skyvern#file-downloading)
### [Authentication](https://github.com/Skyvern-AI/skyvern#authentication)
### [2FA Support (TOTP)](https://github.com/Skyvern-AI/skyvern#--2fa-support-totp)
### [Password Manager Integrations](https://github.com/Skyvern-AI/skyvern#password-manager-integrations)
### [Model Context Protocol (MCP)](https://github.com/Skyvern-AI/skyvern#model-context-protocol-mcp)
### [Zapier / Make.com / N8N Integration](https://github.com/Skyvern-AI/skyvern#zapier--makecom--n8n-integration)

## [Real-world Examples of Skyvern](https://github.com/Skyvern-AI/skyvern#real-world-examples-of-skyvern)

Showcasing how Skyvern is being used in real-world applications, with links to live examples.

## [Contributor Setup](https://github.com/Skyvern-AI/skyvern#contributor-setup)

Instructions for setting up your development environment.

## [Documentation](https://github.com/Skyvern-AI/skyvern#documentation)

Comprehensive documentation is available on the [docs page](https://docs.skyvern.com).

## [Supported LLMs](https://github.com/Skyvern-AI/skyvern#supported-llms)

Details on supported LLMs and environment variables.

## [Feature Roadmap](https://github.com/Skyvern-AI/skyvern#feature-roadmap)

Future developments and planned features.

## [Contributing](https://github.com/Skyvern-AI/skyvern#contributing)

Guidelines for contributing to Skyvern.

## Telemetry

Optional telemetry collection to help us understand usage.

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)