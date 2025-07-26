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
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="Stars"/></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"/></a>
</p>

**Skyvern is a powerful tool that uses Large Language Models (LLMs) and computer vision to automate complex browser-based tasks.** Learn more on the [original repo](https://github.com/Skyvern-AI/skyvern).

**Key Features:**

*   **Automated Workflows:** Create and chain together tasks for complex automation.
*   **AI-Powered Interaction:**  Uses Vision LLMs to navigate websites, even those never seen before.
*   **Resilient to Website Changes:**  Adapts to website layout updates without requiring code modifications.
*   **Data Extraction:** Easily extract data from websites with custom schemas.
*   **Form Filling:** Native support for filling out forms.
*   **File Handling:** Download and upload files to block storage.
*   **Livestreaming:**  Real-time viewport streaming for debugging and monitoring.
*   **Authentication:** Supports various authentication methods, including 2FA.
*   **Zapier/Make.com/N8N Integration:** Connect Skyvern to other apps via integrations.
*   **Model Context Protocol (MCP):** Support any LLM that supports MCP.

**Quickstart:**

1.  **Install:** `pip install skyvern`
2.  **Run:** `skyvern quickstart`
3.  **Run task via UI (recommended):**

```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task

4.  **Run task via code:**

```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```

See the [full Quickstart](https://github.com/Skyvern-AI/skyvern#quickstart) for more details.

**How It Works:**

Skyvern leverages a task-driven autonomous agent design, inspired by projects like BabyAGI and AutoGPT, coupled with browser automation using libraries like Playwright. This allows it to:

*   Comprehend websites and plan actions.
*   Operate on websites without custom code.
*   Adapt to website layout changes.
*   Apply workflows across many sites.
*   Reason through complex scenarios.

See the [How it Works](https://github.com/Skyvern-AI/skyvern#how-it-works) section for more details.

**[Real-world examples of Skyvern](https://github.com/Skyvern-AI/skyvern#real-world-examples-of-skyvern):**

*   Invoice downloading
*   Job application automation
*   Materials procurement
*   Government website navigation
*   Contact form filling
*   Retrieving insurance quotes

**Performance & Evaluation:**

Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai) and is the best-performing agent on WRITE tasks.

**Advanced Usage:**

*   Control your own browser
*   Run Skyvern with any remote browser
*   Get consistent output schema
*   Debugging Commands
*   Docker Compose Setup

See the [Advanced Usage](https://github.com/Skyvern-AI/skyvern#advanced-usage) section for more details.

**[Documentation](https://docs.skyvern.com)**: Find comprehensive guides and references.

**[Contributor Setup](https://github.com/Skyvern-AI/skyvern#contributor-setup)**: Get started with the development environment.

**Supported LLMs:**
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

**Telemetry:**
By default, Skyvern collects basic usage statistics. To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

**[Feature Roadmap](https://github.com/Skyvern-AI/skyvern#feature-roadmap):**  See what's planned for future releases.

**[Contributing](https://github.com/Skyvern-AI/skyvern#contributing):** We welcome contributions!

**[License](https://github.com/Skyvern-AI/skyvern#license):** AGPL-3.0.

**Star History:**
[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)