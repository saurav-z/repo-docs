<h1 align="center">
  Skyvern: Automate Web Workflows with AI 
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

**Skyvern revolutionizes web automation by using Large Language Models (LLMs) and computer vision to intelligently navigate and interact with websites, making traditional, brittle automation a thing of the past.**  Explore the power of Skyvern – automate complex browser-based tasks with ease!  [Visit the original repo](https://github.com/Skyvern-AI/skyvern).

<p align="center">
  <img src="fern/images/geico_shu_recording_cropped.gif" alt="Skyvern in Action"/>
</p>

## Key Features

*   ✨ **LLM-Powered Automation:**  Skyvern leverages the power of LLMs to understand and interact with websites intelligently, eliminating the need for brittle, code-dependent automation.
*   🌐 **Cross-Website Compatibility:**  Operate on websites you've never seen before, adapting to website changes without code modifications.
*   ✅ **Workflow Automation:** Chain multiple tasks together for complex processes, including navigation, data extraction, form filling, and file handling.
*   📺 **Livestreaming & Debugging:**  Watch Skyvern in real-time via a livestream to see exactly what it's doing and intervene when needed.
*   🔑 **Secure Authentication:**  Supports various authentication methods, including 2FA (TOTP), password manager integrations, and more.
*   🛠️ **Flexible Integration:** Integrates with Zapier, Make.com, and N8N to connect Skyvern workflows to other apps.
*   💻 **Cloud & Local Options:**  Utilize Skyvern Cloud for a managed experience or run it locally.
*   📊 **Performance:** Achieving SOTA performance on the [WebBench benchmark](webbench.ai)

## Quickstart

### Installation

```bash
pip install skyvern
```

### Running Skyvern

*   **UI (Recommended):**

    1.  Start the Skyvern service and UI:

    ```bash
    skyvern run all
    ```

    2.  Access the UI in your browser: [http://localhost:8080](http://localhost:8080) and run a task.

*   **Code:**

    ```python
    from skyvern import Skyvern

    skyvern = Skyvern()
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    print(task)
    ```

    Run a task in a browser and view task history at [http://localhost:8080/history](http://localhost:8080/history)

### Advanced Usage

*   **Control your own browser (Chrome):**
    ```python
    from skyvern import Skyvern
    browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" # Example Path
    skyvern = Skyvern(browser_path=browser_path)
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    ```
*   **Run Skyvern with any remote browser:**
    ```python
    from skyvern import Skyvern
    skyvern = Skyvern(cdp_url="your cdp connection url")
    task = await skyvern.run_task(prompt="Find the top post on hackernews today")
    ```
*   **Get consistent output schema:**
    ```python
    from skyvern import Skyvern
    task = await skyvern.run_task(
        prompt="Find the top post on hackernews today",
        data_extraction_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the top post"},
                "url": {"type": "string", "description": "The URL of the top post"},
                "points": {"type": "integer", "description": "Number of points the post has received"}
            }
        }
    )
    ```

*   **Helpful Commands**
```bash
skyvern run server  # Launch Skyvern Server Separately
skyvern run ui    # Launch Skyvern UI
skyvern status    # Check status of the Skyvern service
skyvern stop all  # Stop the Skyvern service
skyvern stop ui   # Stop the Skyvern UI
skyvern stop server   # Stop the Skyvern Server Separately
```

## Docker Compose Setup

See instructions in the original README for Docker installation and setup.

## Skyvern Features

*   **Skyvern Tasks:**  Individual instructions for website interactions.  Requires `url`, `prompt`, and optional `data schema` and `error codes`.
*   **Skyvern Workflows:** Chain tasks for complex automation, including navigation, data extraction, loops, file handling, and email sending.
*   **Livestreaming:** Real-time browser viewport streaming for debugging and understanding.
*   **Form Filling:**  Native form input automation.
*   **Data Extraction:**  Extract structured data with `data_extraction_schema`
*   **File Downloading:** Download and automatically upload files to block storage (if configured).
*   **Authentication:** Supports various authentication methods (2FA, password managers).

## Real-world examples of Skyvern
*   Invoice Downloading
*   Job Application Automation
*   Materials Procurement
*   Government Website Navigation
*   Contact Form Submission
*   Insurance Quote Retrieval

## Contributor Setup
For a complete local environment CLI Installation
```bash
pip install -e .
```
The following command sets up your development environment to use pre-commit (our commit hook handler)
```
skyvern quickstart contributors
```
1.  Start using the UI by navigating to `http://localhost:8080` in your browser.

## Documentation

Extensive documentation is available on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs

See the original README for a full list of supported LLMs.

## Feature Roadmap

*   (As detailed in the original README)

## Contributing

We welcome contributions! See our [contribution guide](CONTRIBUTING.md) and ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Telemetry

Skyvern collects basic usage statistics; opt-out by setting `SKYVERN_TELEMETRY=false`.

## License

Skyvern is licensed under the [AGPL-3.0 License](LICENSE).

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Skyvern-AI/skyvern&type=Date)](https://star-history.com/#Skyvern-AI/skyvern&Date)