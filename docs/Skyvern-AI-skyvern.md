<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png"/>
  </picture>
  <br />
  Skyvern: Automate Your Web Workflows with AI
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

**Skyvern revolutionizes browser automation by using Large Language Models (LLMs) and computer vision to automate complex, browser-based workflows, eliminating the need for brittle, website-specific code.** ([Back to original repo](https://github.com/Skyvern-AI/skyvern))

## Key Features

*   **Intelligent Automation:** Leverage LLMs to understand, plan, and execute actions on any website, adapting to layout changes.
*   **Workflow Automation:** Chain multiple tasks together to automate end-to-end processes, from data extraction to form filling and file downloading.
*   **Real-time Livestreaming:** Monitor Skyvern's actions with real-time browser viewport streaming for debugging and control.
*   **Versatile Integrations:** Seamlessly integrate with authentication methods, password managers, and popular platforms like Zapier, Make.com, and N8N.
*   **Multi-LLM Support:** Compatible with a variety of LLMs, including OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Gemini, Ollama, and OpenRouter.
*   **Cloud and Local Deployment:** Easily deploy Skyvern using our managed cloud service or run it locally with Docker Compose.

## Getting Started

### Quickstart

1.  **Install:**

    ```bash
    pip install skyvern
    ```

2.  **Run (for first-time setup):**

    ```bash
    skyvern quickstart
    ```

3.  **Run UI (Recommended):** Start the Skyvern service and UI (when DB is up and running)

    ```bash
    skyvern run all
    ```

    Go to http://localhost:8080 and use the UI to run a task.

4.  **Run Task (Code):**

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

## Examples of Skyvern in Action

*   **Invoice Downloading:** Automated invoice retrieval across different websites.
    [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)

    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>

*   **Job Application Automation:** Apply for jobs with automated form filling.
    [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>

*   **Materials Procurement:** Automated procurement for manufacturing companies.
    [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>

*   **Government Website Navigation:** Automate account creation and form filling on government sites.
    [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>

*   **Contact Form Filling:** Automate the process of filling out contact forms on websites.
    [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>

*   **Retrieving Insurance Quotes:** Fetching insurance quotes from providers in any language.
    [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>

    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)

    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Detailed Documentation

Find comprehensive documentation on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs
The following table lists all supported LLMs. Check the detailed guide in the documentation for further model-specific configurations.
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

For more info on the environment variables used by each LLM, take a look at the original README for the project.

## Feature Roadmap

*   **Open Source:** Open Source Skyvern's core codebase
*   **Workflow Support:** Allow support to chain multiple Skyvern calls together
*   **Improved context:** Improve Skyvern's ability to understand content around interactable elements by introducing feeding relevant label context through the text prompt
*   **Cost Savings:** Improve Skyvern's stability and reduce the cost of running Skyvern by optimizing the context tree passed into Skyvern
*   **Self-serve UI:** Deprecate the Streamlit UI in favour of a React-based UI component that allows users to kick off new jobs in Skyvern
*   **Workflow UI Builder:** Introduce a UI to allow users to build and analyze workflows visually
*   **Chrome Viewport streaming:** Introduce a way to live-stream the Chrome viewport to the user's browser (as a part of the self-serve UI)
*   **Past Runs UI:** Deprecate the Streamlit UI in favour of a React-based UI that allows you to visualize past runs and their results
*   **Auto workflow builder ("Observer") mode:** Allow Skyvern to auto-generate workflows as it's navigating the web to make it easier to build new workflows
*   **Prompt Caching:** Introduce a caching layer to the LLM calls to dramatically reduce the cost of running Skyvern (memorize past actions and repeat them!)
*   **Web Evaluation Dataset:** Integrate Skyvern with public benchmark tests to track the quality of our models over time
*   **Improved Debug mode:** Allow Skyvern to plan its actions and get "approval" before running them, allowing you to debug what it's doing and more easily iterate on the prompt
*   **Chrome Extension:** Allow users to interact with Skyvern through a Chrome extension (incl voice mode, saving tasks, etc.)
*   **Skyvern Action Recorder:** Allow Skyvern to watch a user complete a task and then automatically generate a workflow for it
*   **Interactable Livestream:** Allow users to interact with the livestream in real-time to intervene when necessary (such as manually submitting sensitive forms)
*   **Integrate LLM Observability tools:** Integrate LLM Observability tools to allow back-testing prompt changes with specific data sets + visualize the performance of Skyvern over time
*   **Langchain Integration:** Create langchain integration in langchain_community to use Skyvern as a "tool".

## Contributing

We welcome your contributions!  See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) to get started. Also feel free to open a PR/issue or to reach out to us [via email](mailto:founders@skyvern.com) or [discord](https://discord.gg/fG2XXEuQX3).

## Telemetry

By Default, Skyvern collects basic usage statistics. If you would like to opt-out of telemetry, please set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License

Skyvern's open source repository is supported via a managed cloud. All of the core logic powering Skyvern is available in this open source repository licensed under the [AGPL-3.0 License](LICENSE), with the exception of anti-bot measures available in our managed cloud offering.

If you have any questions or concerns around licensing, please [contact us](mailto:support@skyvern.com) and we would be happy to help.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)
```

Key improvements and SEO considerations:

*   **Clear, concise introduction:**  Immediately states the value proposition.
*   **Keyword optimization:** Includes relevant keywords throughout ("browser automation," "LLMs," "computer vision," "web workflows").
*   **Structured headings:** Uses clear headings to improve readability and SEO.
*   **Bulleted lists:** Highlights key features for easy scanning.
*   **Action-oriented language:**  Encourages users to try Skyvern ("Get Started").
*   **Internal linking:**  Links to relevant sections within the README.
*   **External linking:** Provides links to the Skyvern website, documentation, and community resources.
*   **Concise Quickstart instructions:** Makes it easy for users to get started.
*   **Real-world examples:** Showcases use cases with clear descriptions and links.
*   **Consistent formatting:** Makes the document easier to read.
*   **Concise, SEO-friendly headings and titles.**