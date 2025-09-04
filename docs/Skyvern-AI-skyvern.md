<h1 align="center">
  Skyvern: Automate Browser Workflows with AI 🤖
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

**Skyvern is an AI-powered automation platform that enables you to automate complex browser-based tasks with ease.**  Automate repetitive tasks, streamline workflows, and replace brittle automation solutions with the power of LLMs and computer vision.

[Explore the Skyvern repository on GitHub](https://github.com/Skyvern-AI/skyvern).

## Key Features:

*   **AI-Driven Automation:** Utilize Large Language Models (LLMs) and computer vision to understand and interact with websites.
*   **Workflow Automation:** Chain multiple tasks together to automate complex processes.
*   **Web Automation:**
    *   Form Filling: Populate forms automatically.
    *   Data Extraction: Extract structured data from web pages.
    *   File Downloading: Download files directly from websites.
    *   Authentication: Securely handle logins with various methods, including 2FA (TOTP, email, SMS).
*   **Real-time Monitoring:** Live stream the browser view for debugging and oversight.
*   **Flexible Integration:**
    *   Model Context Protocol (MCP) support for diverse LLMs.
    *   Integrations with Zapier, Make.com, and N8N for seamless connectivity.
*   **Performance & Evaluation:** SOTA performance on the [WebBench benchmark](webbench.ai).

## Getting Started

### 1. Install Skyvern

```bash
pip install skyvern
```

### 2. Run Skyvern

```bash
skyvern quickstart
```

### 3. Run Task (UI or Code)

*   **UI (Recommended):** Start the service and UI:

    ```bash
    skyvern run all
    ```
    Then go to http://localhost:8080 and use the UI.
*   **Code:**

    ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
    ```

    You can also configure Skyvern to use either local or cloud services via parameters.

## Advanced Usage

*   **Custom Chrome Browser**: Use your own browser instance and profile.
*   **Remote Browsers**: Connect to browsers via CDP URL.
*   **Consistent Output Schemas**: Define data extraction schemas for predictable results.
*   **Debugging Tools**: Easily manage and debug Skyvern server and UI.

[Refer to the original README for a more detailed breakdown of all the commands and usage instructions.](https://github.com/Skyvern-AI/skyvern)

## Real-World Examples

*   Invoice Downloading across various websites
*   Automated Job Application Process
*   Materials Procurement automation
*   Government Website Automation (account creation, form filling)
*   Contact Form Automation
*   Retrieving Insurance Quotes

## Documentation

Comprehensive documentation can be found on our [📕 docs page](https://docs.skyvern.com).

## Contributing

We welcome contributions! Please see our [contribution guide](CONTRIBUTING.md) and
["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) for ways to contribute.

## Telemetry

Skyvern collects basic usage statistics to help improve the platform (opt-out via `SKYVERN_TELEMETRY=false`).

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).