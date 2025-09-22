<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png"/>
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo"/>
  </picture>
  <br/>
  Skyvern: Automate Your Browser Workflows with AI
</h1>

<p align="center">
  <strong>Effortlessly automate browser-based tasks using LLMs and computer vision, without writing custom scripts.</strong>
</p>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black"/></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black"/></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord"/></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" /></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern"/></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social"/></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin"/></a>
</p>

[Skyvern](https://github.com/Skyvern-AI/skyvern) revolutionizes browser automation by leveraging Large Language Models (LLMs) and computer vision. Say goodbye to brittle, website-specific scripts and embrace intelligent, adaptable automation.  Visit the [original repo](https://github.com/Skyvern-AI/skyvern) for the source code and to contribute.

**Key Features:**

*   ✅ **Intelligent Automation:** Uses LLMs to understand and interact with websites, eliminating the need for hardcoded selectors.
*   ✅ **Website Agnostic:** Works on websites it's never seen before, adapting to layout changes automatically.
*   ✅ **Workflow Automation:**  Create complex, multi-step workflows for tasks like data extraction, form filling, and file downloading.
*   ✅ **SOTA Performance:** Achieves industry-leading performance on the WebBench benchmark, particularly on WRITE tasks.
*   ✅ **Cloud & Local Deployment:** Easily run Skyvern with both the managed cloud offering or locally.
*   ✅ **Advanced Features:** Support for 2FA, password manager integrations, model context protocol (MCP), and integrations with Zapier, Make.com, and N8N.
*   ✅ **Livestreaming:** Monitor browser interactions in real-time.

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern Demo"/>
</p>

## How Skyvern Works

Skyvern employs a swarm of intelligent agents, inspired by systems like BabyAGI and AutoGPT, to understand, plan, and execute actions on websites. This approach offers key advantages:

*   **Adaptability:**  Navigates websites without custom code, mapping visual elements to actions.
*   **Resilience:**  Resistant to website layout changes, using LLMs to interpret the visual elements.
*   **Scalability:**  Applies a single workflow across many websites, reasoning through necessary interactions.
*   **Intelligent Decision-Making:** Leverages LLMs to handle complex scenarios, like inferring information and handling nuanced data.

A detailed technical report can be found [here](https://blog.skyvern.com/skyvern-2-0-state-of-the-art-web-navigation-with-85-8-on-webvoyager-eval/).

## Quickstart

### Install & Run

**Prerequisites:**

*   [Python 3.11.x](https://www.python.org/downloads/) (works with 3.12, not yet 3.13)
*   [NodeJS & NPM](https://nodejs.org/en/download/)

**For Windows:**

*   [Rust](https://rustup.rs/)
*   VS Code with C++ dev tools and Windows SDK

**Steps:**

1.  **Install Skyvern:**

    ```bash
    pip install skyvern
    ```

2.  **Quickstart (for initial setup):**

    ```bash
    skyvern quickstart
    ```

3.  **Run Tasks (UI Recommended):**

    ```bash
    skyvern run all
    ```

    Then access the UI at http://localhost:8080 to create and run tasks.

4.  **Run Tasks (Code):**

    ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
    ```

    Customize your connection type with `api_key`, `base_url`, `cdp_url` and `browser_path`.

## Performance and Evaluation

Skyvern is a top performer, demonstrating high accuracy on the [WebBench benchmark](webbench.ai).  It is particularly effective in WRITE tasks.

*   **WRITE Tasks Performance:** Skyvern excels in form filling, logging in, and downloading files.

<p align="center">
  <img src="fern/images/performance/webbench_write.png" alt="Webbench Write Performance"/>
</p>

## Skyvern Features (Detailed)

### Skyvern Tasks
Tasks are a single request to Skyvern, instructing it to navigate through a website and accomplish a specific goal.

### Skyvern Workflows
Workflows are a way to chain multiple tasks together to form a cohesive unit of work.

<p align="center">
  <img src="fern/images/block_example_v2.png" alt="Workflows"/>
</p>

##  More Features

*   **Livestreaming:** Real-time browser viewport streaming for debugging.
*   **Form Filling:** Native form input capabilities.
*   **Data Extraction:** Extract structured data from websites.
*   **File Downloading:** Automatic file downloads and upload to block storage.
*   **Authentication:** Supports a number of authentication methods to make it easier to automate tasks behind a login.
    *   🔐 2FA Support (TOTP)
    *   Password Manager Integrations
*   **Model Context Protocol (MCP):** Supports any LLM supporting MCP.
*   **Zapier / Make.com / N8N Integration:** Connect Skyvern with other apps.

## Real-world examples of Skyvern

*   [Book a demo to see it live](https://meetings.hubspot.com/skyvern/demo)
    Invoice Downloading on many different websites
    <p align="center">
      <img src="fern/images/invoice_downloading.gif"/>
    </p>
*   [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
    Automate the job application process
    <p align="center">
      <img src="fern/images/job_application_demo.gif"/>
    </p>
*   [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
    Automate materials procurement for a manufacturing company
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif"/>
    </p>
*   [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
    Navigating to government websites to register accounts or fill out forms
    <p align="center">
      <img src="fern/images/edd_services.gif"/>
    </p>
*   [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    Filling out random contact us forms
    <p align="center">
      <img src="fern/images/contact_forms.gif"/>
    </p>
*   [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    Retrieving insurance quotes from insurance providers in any language
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif"/>
    </p>
*   [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif"/>
    </p>

## Supported LLMs

[See supported LLMs](https://github.com/Skyvern-AI/skyvern#supported-llms) and their environment variables in the original README.

## Feature Roadmap

[See the feature roadmap](https://github.com/Skyvern-AI/skyvern#feature-roadmap) in the original README.

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md) and [Help Wanted issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Telemetry

By default, Skyvern collects basic usage statistics.  Opt-out by setting `SKYVERN_TELEMETRY=false`.

## License

Skyvern is open source, licensed under the [AGPL-3.0 License](LICENSE), with the exception of anti-bot measures in the managed cloud. Contact us at [support@skyvern.com](mailto:support@skyvern.com) for any licensing questions.