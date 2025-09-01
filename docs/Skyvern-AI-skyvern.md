<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  Skyvern: Automate Browser Tasks with the Power of AI
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

**Skyvern empowers you to automate complex browser workflows with the intelligence of Large Language Models (LLMs) and computer vision, eliminating the need for brittle, website-specific scripts.** ([View on GitHub](https://github.com/Skyvern-AI/skyvern))

## Key Features

*   **AI-Powered Automation:** Leverage LLMs and computer vision to navigate and interact with websites you've never seen before.
*   **Robust to Website Changes:**  Skyvern adapts to website updates without requiring code modifications.
*   **Cross-Website Automation:**  Apply the same workflow across numerous websites with ease.
*   **Advanced Reasoning:**  Handle complex scenarios through LLM-powered reasoning.
*   **Workflows:** Chain multiple tasks together to automate more complex processes.
*   **Data Extraction:**  Extract structured data from websites using schema definitions.
*   **File Downloading & Uploading:**  Download files and upload them to block storage.
*   **2FA Support:** Automate workflows requiring 2FA (QR, Email, SMS).
*   **Integrations:** Integrate with Zapier, Make.com, and N8N.
*   **Livestreaming:** See the browser in real-time.

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
Go to http://localhost:8080 to run a task

#### Code
```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

Skyvern starts running the task in a browser that pops up and closes it when the task is done. You will be able to view the task from http://localhost:8080/history

## How It Works

Skyvern leverages a swarm of AI agents to understand websites, plan actions, and execute tasks using browser automation. It overcomes the limitations of traditional automation by using Vision LLMs to map visual elements to actions, making it resistant to website changes and adaptable to new sites.

[See a system diagram](fern/images/skyvern_2_0_system_diagram.png)

For a detailed technical report, visit [Skyvern 2.0 Technical Report](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

<img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo">

## Performance & Evaluation

Skyvern achieves state-of-the-art (SOTA) performance on the [WebBench benchmark](webbench.ai) with a 64.4% accuracy, with 74% accuracy on tasks requiring writing.

[See Skyvern's WebBench results](fern/images/performance/webbench_overall.png)
[See Skyvern's WebBench WRITE results](fern/images/performance/webbench_write.png)

## Advanced Usage

### Control Your Own Chrome Browser

*   [Instructions and code snippets](https://github.com/Skyvern-AI/skyvern#control-your-own-browser-chrome)

### Run Skyvern with any remote browser
*   [Instructions and code snippets](https://github.com/Skyvern-AI/skyvern#run-skyvern-with-any-remote-browser)

### Consistent Output Schema

*   [Instructions and code snippets](https://github.com/Skyvern-AI/skyvern#get-consistent-output-schema-from-your-run)

### Debugging commands

*   [Helpful commands to debug issues](https://github.com/Skyvern-AI/skyvern#helpful-commands-to-debug-issues)

### Docker Compose Setup

*   [Instructions for Docker Compose](https://github.com/Skyvern-AI/skyvern#docker-compose-setup)

## Skyvern Features

### Skyvern Tasks

Tasks are the fundamental building blocks of Skyvern. Specify a URL, prompt, and optional data schema for your task.

[See screenshot](fern/images/skyvern_2_0_screenshot.png)

### Skyvern Workflows

Chain multiple tasks together to create automated workflows.

[See an example](fern/images/invoice_downloading_workflow_example.png)

### Additional Features

*   **Livestreaming:** See the browser in real-time.
*   **Form Filling:** Native form filling capabilities.
*   **Data Extraction:** Extract structured data.
*   **File Downloading:** Download files and upload them to block storage.
*   **Authentication:** Supports various authentication methods, including 2FA.
    *   🔐 Learn more about 2FA support [here](https://docs.skyvern.com/credentials/totp).
    *   **Password Manager Integrations:**
        *   [x] Bitwarden
        *   [ ] 1Password
        *   [ ] LastPass
*   **Model Context Protocol (MCP):** Integrates with LLMs supporting MCP.  See the MCP documentation [here](https://github.com/Skyvern-AI/skyvern/blob/main/integrations/mcp/README.md)
*   **Zapier / Make.com / N8N Integration**
    *   * [Zapier](https://docs.skyvern.com/integrations/zapier)
    *   * [Make.com](https://docs.skyvern.com/integrations/make.com)
    *   * [N8N](https://docs.skyvern.com/integrations/n8n)

## Real-World Examples

*   Invoice Downloading on many different websites
    [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)
    [See a demo](fern/images/invoice_downloading.gif)
*   Automate the job application process
    [See it in action](https://app.skyvern.com/tasks/create/job_application)
    [See a demo](fern/images/job_application_demo.gif)
*   Automate materials procurement for a manufacturing company
    [See it in action](https://app.skyvern.com/tasks/create/finditparts)
    [See a demo](fern/images/finditparts_recording_crop.gif)
*   Navigating to government websites to register accounts or fill out forms
    [See it in action](https://app.skyvern.com/tasks/create/california_edd)
    [See a demo](fern/images/edd_services.gif)
*   Filling out random contact us forms
    [See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    [See a demo](fern/images/contact_forms.gif)
*   Retrieving insurance quotes from insurance providers in any language
    [See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    [See a demo](fern/images/bci_seguros_recording.gif)
    [See it in action](https://app.skyvern.com/tasks/create/geico)
    [See a demo](fern/images/geico_shu_recording_cropped.gif)

## Contributor Setup

For a complete local environment CLI Installation, see:
```bash
pip install -e .
```
The following command sets up your development environment to use pre-commit (our commit hook handler):
```
skyvern quickstart contributors
```
[Instructions for UI setup](https://github.com/Skyvern-AI/skyvern#contributor-setup)

## Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

[See supported LLMs and required Environment Variables](https://github.com/Skyvern-AI/skyvern#supported-llms)

## Feature Roadmap

*   [x] Open Source
*   [x] Workflow support
*   [x] Improved context
*   [x] Cost Savings
*   [x] Self-serve UI
*   [x] Workflow UI Builder
*   [x] Chrome Viewport streaming
*   [x] Past Runs UI
*   [X] Auto workflow builder ("Observer") mode
*   [x] Prompt Caching
*   [x] Web Evaluation Dataset
*   [ ] Improved Debug mode
*   [ ] Chrome Extension
*   [ ] Skyvern Action Recorder
*   [ ] Interactable Livestream
*   [ ] Integrate LLM Observability tools
*   [x] Langchain Integration

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md) and "Help Wanted" issues. Contact us via [email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).
[Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme) is available for project overview, build, and usage questions.

## Telemetry

Skyvern collects basic usage statistics by default. Opt-out by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE).  Anti-bot measures in our managed cloud offering are not covered by this license.  Contact us at [support@skyvern.com](mailto:support@skyvern.com) for any licensing questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)