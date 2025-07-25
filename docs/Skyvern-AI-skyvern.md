<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br />
  Skyvern: Automate Browser Workflows with AI 🐉
</h1>

<p align="center">
  <strong>Effortlessly automate complex browser-based tasks using the power of Large Language Models (LLMs) and Computer Vision.</strong>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

[Skyvern](https://github.com/Skyvern-AI/skyvern) allows you to automate complex browser-based workflows, overcoming the limitations of traditional, brittle automation methods. Skyvern utilizes the latest advancements in LLMs and computer vision to understand and interact with web pages, making automation robust and adaptable to website changes.

## Key Features

*   **AI-Powered Automation:** Uses LLMs and computer vision to interact with websites, adapting to changes without code modification.
*   **No Code Required:**  Automate workflows by describing your tasks with natural language.
*   **Web Application Automation:** From filling out forms to downloading files, Skyvern automates a wide range of tasks.
*   **Robustness:** Designed to handle website layout changes, ensuring workflows remain functional.
*   **Workflow Creation:** Chain multiple tasks together to create complex automated workflows.
*   **Form Filling:** Skyvern can intelligently fill form inputs on web pages.
*   **Data Extraction:** Extract structured data from websites using a specified schema.
*   **File Downloading:** Easily download files directly from websites.
*   **Real-time Monitoring:** Stream browser viewports live to see exactly what Skyvern is doing on the web
*   **Authentication Support:** Includes password manager integrations and 2FA support (including QR-based, email, and SMS).
*   **Integration Options**: Integrate with Zapier, Make.com, N8N, and more.
*   **Model Context Protocol (MCP) Support:** Integrate with LLMs that support MCP.

## Getting Started

### Install Skyvern

```bash
pip install skyvern
```

### Run Skyvern

```bash
skyvern quickstart
```

### Run Task via UI

```bash
skyvern run all
```

Then, go to http://localhost:8080 and use the UI to run a task

### Run Task via Code

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

## Advanced Usage

See the original README for detailed instructions on:

*   Running tasks on Skyvern Cloud
*   Running Skyvern locally
*   Using the UI
*   Controlling Your Browser

## How It Works

Skyvern operates using a swarm of AI agents, inspired by BabyAGI and AutoGPT, and leverages browser automation libraries like Playwright to interact with websites.  This approach provides:

*   Adaptability to previously unseen websites.
*   Resistance to website layout changes.
*   The ability to apply a single workflow across many websites.
*   LLM-powered reasoning to address complex situations.

See the system diagram in the original README.  A detailed technical report can be found [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Demo

See Skyvern in action!  Check out the demos in the original README.

## Performance & Evaluation

Skyvern achieves state-of-the-art (SOTA) performance. The full evaluation can be found [here](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)

## Feature Roadmap

See the original README for feature roadmaps.

## Documentation

For detailed documentation, visit our [docs page](https://docs.skyvern.com).

## Supported LLMs

See the original README for supported LLMs and environment variables.

## Contributing

We welcome contributions! Please refer to the [contribution guide](CONTRIBUTING.md) and the ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started.

## Telemetry

See the original README for information on telemetry.

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE), see the original README for more information.

## Star History

See the original README for star history.