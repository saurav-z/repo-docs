<h1 align="center">
  <a href="https://www.skyvern.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
      <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
    </picture>
  </a>
  <br>
  Automate Your Browser Workflows with AI: Unleash the Power of Skyvern!
  <br>
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"></a>
</p>

[Skyvern](https://github.com/Skyvern-AI/skyvern) revolutionizes browser automation by using Large Language Models (LLMs) and computer vision to automate complex web-based workflows.  Tired of brittle and unreliable automation solutions? Skyvern provides a robust API for automating any browser-based task.

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Automation Demo">
</p>

## Key Features of Skyvern:

*   **LLM-Powered Automation:** Utilize LLMs to understand, plan, and execute tasks on websites, eliminating the need for website-specific code.
*   **Resilient to Website Changes:**  Skyvern adapts to website layout changes, ensuring your automation remains functional.
*   **Cross-Website Compatibility:**  Apply the same workflow across numerous websites for scalable automation.
*   **Advanced Reasoning:** Leverages LLMs to handle complex scenarios, such as inferring answers and understanding nuanced product variations.
*   **[Tasks and Workflows](https://docs.skyvern.com/skyvern-features/skyvern-tasks)**: Build and orchestrate complex automations using tasks, workflows, and advanced features like data extraction, file downloading, and authentication.
*   **[Livestreaming](https://docs.skyvern.com/skyvern-features/livestreaming)**: Watch Skyvern in action with a live viewport stream of your browser.
*   **[2FA and Password Manager Support](https://docs.skyvern.com/credentials/totp)**: Automate workflows that require 2FA and password management.
*   **[Integration](https://docs.skyvern.com/integrations)**: Integrate with Zapier, Make.com and N8N for extended capabilities.
*   **SOTA Performance**: Skyvern performs at a State of the Art on the WebBench benchmark.

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
Start the Skyvern service and UI

```bash
skyvern run all
```

Go to http://localhost:8080 and use the UI to run a task.

#### Code
```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```
Skyvern starts running the task in a browser that pops up and closes it when the task is done. You will be able to view the task from http://localhost:8080/history

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

### Additional Usage

*   **[How it Works](https://docs.skyvern.com/how-it-works)**: Skyvern's architecture, inspired by BabyAGI and AutoGPT, uses a swarm of agents for web interaction, planning, and execution.
*   **[Performance & Evaluation](https://blog.skyvern.com/web-bench-a-new-way-to-compare-ai-browser-agents/)**: Explore the SOTA performance and detailed benchmarks.
*   **[Advanced Usage](https://github.com/Skyvern-AI/skyvern#advanced-usage)**:  Control your browser, use a remote browser,  get consistent output schema, and debug issues.
*   **[Docker Compose Setup](https://github.com/Skyvern-AI/skyvern#docker-compose-setup)**: Get started with Docker and Skyvern.

###  [Real-World Examples](https://github.com/Skyvern-AI/skyvern#real-world-examples-of-skyvern)

*   Invoice Downloading
*   Job Application Automation
*   Material Procurement for Manufacturing
*   Government Website Automation
*   Contact Form Filling
*   Insurance Quote Retrieval

## Documentation

For comprehensive guidance, consult our [docs page](https://docs.skyvern.com).

## Supported LLMs

| Provider    | Supported Models                                          |
| :---------- | :-------------------------------------------------------- |
| OpenAI      | gpt4-turbo, gpt-4o, gpt-4o-mini                             |
| Anthropic   | Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet)       |
| Azure OpenAI | Any GPT models. Better performance with a multimodal llm (azure/gpt4-o)                                                |
| AWS Bedrock | Anthropic Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 (Sonnet) |
| Gemini      | Gemini 2.5 Pro and flash, Gemini 2.0                        |
| Ollama      | Run any locally hosted model via [Ollama](https://github.com/ollama/ollama)          |
| OpenRouter  | Access models through [OpenRouter](https://openrouter.ai)      |
| OpenAI-compatible | Any custom API endpoint that follows OpenAI's API format (via [liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible))               |

#### [Environment Variables](https://github.com/Skyvern-AI/skyvern#environment-variables)

## Feature Roadmap

*   [x] **Open Source** - Open Source Skyvern's core codebase
*   [x] **Workflow support** - Allow support to chain multiple Skyvern calls together
*   [x] **Improved context** - Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
*   [x] **Cost Savings** - Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
*   [x] **Self-serve UI** - Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
*   [x] **Workflow UI Builder** - Introduce a UI to allow users to build and analyze workflows visually
*   [x] **Chrome Viewport streaming** - Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
*   [x] **Past Runs UI** - Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
*   [X] **Auto workflow builder ("Observer") mode** - Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
*   [x] **Prompt Caching** - Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
*   [x] **Web Evaluation Dataset** - Integrate Skyvern with public benchmark tests to track the quality of our models over time
*   [ ] **Improved Debug mode** - Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
*   [ ] **Chrome Extension** - Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
*   [ ] **Skyvern Action Recorder** - Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
*   [ ] **Interactable Livestream** - Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
*   [ ] **Integrate LLM Observability tools** - Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
*   [x] **Langchain Integration** - Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We value contributions!  Refer to our [contribution guide](CONTRIBUTING.md) and [Help Wanted issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started. Consider using [Code Sage](https://sage.storia.ai?utm_source=github&utm_medium=referral&utm_campaign=skyvern-readme) to get a high level overview of the project.

## Telemetry

Opt-out of telemetry by setting the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Licensed under the [AGPL-3.0 License](LICENSE).

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)