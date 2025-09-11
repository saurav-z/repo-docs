<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br/>
  Skyvern: Automate Browser Workflows with the Power of AI
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

**Skyvern empowers you to automate complex browser-based workflows using Large Language Models (LLMs) and computer vision, eliminating the need for brittle, website-specific scripts.**

Explore the original repository at [https://github.com/Skyvern-AI/skyvern](https://github.com/Skyvern-AI/skyvern).

## Key Features

*   ✅ **Automated Browser Automation:** Automate complex browser tasks without custom code.
*   ✅ **LLM-Powered Interactions:** Leverages LLMs for intelligent website navigation and interaction.
*   ✅ **Website Agnostic:** Works on websites it's never seen before, adapting to changes.
*   ✅ **Workflow Automation:** Chain multiple tasks into sophisticated automated workflows.
*   ✅ **Data Extraction:** Extract structured data from websites with schema support.
*   ✅ **Form Filling:** Automatically fill out forms on any website.
*   ✅ **File Downloading:** Download files directly from websites.
*   ✅ **Livestreaming:** Watch Skyvern's actions live in your browser.
*   ✅ **2FA and Password Manager Integrations:** Integrations for secure automation.
*   ✅ **Integrations:** Supported integrations for Zapier, Make.com, and N8N.

## Quickstart

### 1. Install

```bash
pip install skyvern
```

### 2. Run with UI

```bash
skyvern run all
```

Navigate to http://localhost:8080 to use the UI and run your tasks.

### 3. Run with Code
```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## How it Works

Skyvern uses a swarm of AI agents, inspired by BabyAGI and AutoGPT, to:

1.  **Comprehend** the website's structure.
2.  **Plan** the necessary actions.
3.  **Execute** those actions using browser automation libraries like Playwright.

This approach offers several advantages:

*   **Adaptability:** Handles websites it hasn't seen before.
*   **Resilience:** Resistant to website layout changes.
*   **Scalability:** Applies workflows across numerous websites.
*   **Intelligence:** Uses LLMs to reason through complex scenarios.

## Performance & Evaluation
Skyvern has SOTA performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy.

See the detailed technical report and evaluation [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

### WRITE Tasks
Skyvern performs best on WRITE tasks such as form filling and downloading files.

## Advanced Usage

### 1. Control your own browser (Chrome)
  *   Just with Python Code
  *   With Skyvern Service

### 2. Run Skyvern with any remote browser

### 3. Get consistent output schema from your run

### 4. Helpful commands to debug issues

### Docker Compose setup

## Skyvern Features

### 1. Skyvern Tasks
### 2. Skyvern Workflows
### 3. Livestreaming
### 4. Form Filling
### 5. Data Extraction
### 6. File Downloading
### 7. Authentication

## Real-World Examples

*   Invoice Downloading (see it live)
*   Job Application Automation
*   Materials Procurement for Manufacturing
*   Government Website Automation
*   Contact Us Form Filling
*   Insurance Quote Retrieval

## Documentation

Find extensive documentation on our [docs page](https://docs.skyvern.com).

## Supported LLMs

[See Table in Original README for more details.]

## Feature Roadmap

*   [Open Source](x)
*   [Workflow support](x)
*   [Improved context](x)
*   [Cost Savings](x)
*   [Self-serve UI](x)
*   [Workflow UI Builder](x)
*   [Chrome Viewport streaming](x)
*   [Past Runs UI](x)
*   [Auto workflow builder ("Observer") mode](x)
*   [Prompt Caching](x)
*   [Web Evaluation Dataset](x)
*   [Improved Debug mode]
*   [Chrome Extension]
*   [Skyvern Action Recorder]
*   [Interactable Livestream]
*   [Integrate LLM Observability tools]
*   [Langchain Integration](x)

## Contributing

We welcome contributions! See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started.

## Telemetry

Opt-out of telemetry by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).

## Star History

[Include Star History Chart Here - Instructions in Original]