<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fern/images/skyvern_logo.png">
    <img height="120" src="fern/images/skyvern_logo_blackbg.png" alt="Skyvern Logo">
  </picture>
  <br>
  Skyvern: Automate Browser Workflows with the Power of LLMs
</h1>

<p align="center">
  <a href="https://www.skyvern.com/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&logoColor=black" alt="Website"></a>
  <a href="https://docs.skyvern.com/"><img src="https://img.shields.io/badge/Docs-yellow?logo=gitbook&logoColor=black" alt="Documentation"></a>
  <a href="https://discord.gg/fG2XXEuQX3"><img src="https://img.shields.io/discord/1212486326352617534?logo=discord&label=discord" alt="Discord"></a>
  <a href="https://github.com/skyvern-ai/skyvern"><img src="https://img.shields.io/github/stars/skyvern-ai/skyvern" alt="GitHub Stars"></a>
  <a href="https://github.com/Skyvern-AI/skyvern/blob/main/LICENSE"><img src="https://img.shields.io/github/license/skyvern-ai/skyvern" alt="License"></a>
  <a href="https://twitter.com/skyvernai"><img src="https://img.shields.io/twitter/follow/skyvernai?style=social" alt="Twitter"></a>
  <a href="https://www.linkedin.com/company/95726232"><img src="https://img.shields.io/badge/Follow%20 on%20LinkedIn-8A2BE2?logo=linkedin" alt="LinkedIn"></a>
</p>

Skyvern empowers you to automate complex browser-based tasks effortlessly, using the combined intelligence of Large Language Models (LLMs) and computer vision.  [Explore the Skyvern repository](https://github.com/Skyvern-AI/skyvern) to unlock the future of web automation.

## Key Features

*   **LLM-Powered Automation:** Automate web tasks by leveraging the understanding and reasoning capabilities of LLMs.
*   **Computer Vision Integration:** Interact with websites through computer vision, allowing for robust and flexible automation.
*   **Resilient Automation:**  Navigate through website changes with ease, thanks to a design that isn't reliant on brittle XPath or DOM-based selectors.
*   **Workflow Automation:** Chain multiple tasks together to accomplish larger, more complex automation goals.
*   **Form Filling & Data Extraction:** Easily fill out forms and extract data from websites.
*   **2FA Support:** Automate tasks behind logins requiring 2FA.
*   **Livestreaming:** Observe Skyvern's actions in real-time with viewport streaming.
*   **Multi-LLM Support:** Supports OpenAI, Anthropic, Gemini, Ollama, and more.

## Quickstart

### 1. Install Skyvern
```bash
pip install skyvern
```

### 2. Run Skyvern (choose one):

#### a) Using the UI (Recommended)
```bash
skyvern run all
```
Then navigate to http://localhost:8080 to run tasks.

#### b) Using Code
```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```
Runs the task in a browser, which then closes automatically. View task results at http://localhost:8080/history.

### 3. Run tasks on different targets:
```python
from skyvern import Skyvern

# Run on Skyvern Cloud (requires API key)
skyvern = Skyvern(api_key="SKYVERN API KEY")

# Local Skyvern service (requires base URL)
skyvern = Skyvern(base_url="http://localhost:8000", api_key="LOCAL SKYVERN API KEY")

task = await skyvern.run_task(prompt="Find the top post on hackernews today")
print(task)
```
## Real-World Examples

See how Skyvern is transforming web automation:

*   **Invoice Downloading:** Automate invoice downloads across various websites.
    Book a demo to see it live: [Book a demo](https://meetings.hubspot.com/skyvern/demo)
    <p align="center">
      <img src="fern/images/invoice_downloading.gif" alt="Invoice Downloading Demo"/>
    </p>

*   **Job Application Automation:** Streamline your job application process.
    [💡 See it in action](https://app.skyvern.com/tasks/create/job_application)
    <p align="center">
      <img src="fern/images/job_application_demo.gif" alt="Job Application Demo"/>
    </p>

*   **Material Procurement:** Automate materials procurement for manufacturing.
    [💡 See it in action](https://app.skyvern.com/tasks/create/finditparts)
    <p align="center">
      <img src="fern/images/finditparts_recording_crop.gif" alt="Material Procurement Demo"/>
    </p>

*   **Government Website Navigation:** Register accounts and fill out forms on government websites.
    [💡 See it in action](https://app.skyvern.com/tasks/create/california_edd)
    <p align="center">
      <img src="fern/images/edd_services.gif" alt="Government Website Demo"/>
    </p>

*   **Contact Form Filling:** Automate contact form submissions.
    [💡 See it in action](https://app.skyvern.com/tasks/create/contact_us_forms)
    <p align="center">
      <img src="fern/images/contact_forms.gif" alt="Contact Form Demo"/>
    </p>

*   **Insurance Quote Retrieval:** Get insurance quotes from different providers in any language.
    [💡 See it in action](https://app.skyvern.com/tasks/create/bci_seguros)
    <p align="center">
      <img src="fern/images/bci_seguros_recording.gif" alt="Insurance Quote Demo"/>
    </p>
    [💡 See it in action](https://app.skyvern.com/tasks/create/geico)
    <p align="center">
      <img src="fern/images/geico_shu_recording_cropped.gif" alt="Geico Demo"/>
    </p>
## Performance & Evaluation

Skyvern excels in web automation:

*   **WebBench Benchmark:** Skyvern achieves SOTA performance on the [WebBench benchmark](webbench.ai) with 64.4% accuracy.
    <p align="center">
      <img src="fern/images/performance/webbench_overall.png" alt="WebBench Overall Performance"/>
    </p>

*   **WRITE Tasks:** Skyvern leads in WRITE task performance (RPA-adjacent tasks).
    <p align="center">
      <img src="fern/images/performance/webbench_write.png" alt="WebBench Write Task Performance"/>
    </p>

## Advanced Usage
*(See original README for complete examples)*
### Control your own browser (Chrome)
```python
from skyvern import Skyvern

# The path to your Chrome browser. This example path is for Mac.
browser_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
skyvern = Skyvern(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY",
    browser_path=browser_path,
)
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Run Skyvern with any remote browser
```python
from skyvern import Skyvern

skyvern = Skyvern(cdp_url="your cdp connection url")
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
)
```

### Get consistent output schema from your run
```python
from skyvern import Skyvern

skyvern = Skyvern()
task = await skyvern.run_task(
    prompt="Find the top post on hackernews today",
    data_extraction_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the top post"
            },
            "url": {
                "type": "string",
                "description": "The URL of the top post"
            },
            "points": {
                "type": "integer",
                "description": "Number of points the post has received"
            }
        }
    }
)
```

### Helpful commands to debug issues
*(See original README for a complete list)*
```bash
# Launch the Skyvern Server Separately*
skyvern run server
```

## Docker Compose setup
*(See original README for full instructions)*
```bash
docker compose up -d
```
Navigate to `http://localhost:8080`

## Documentation

Find in-depth information on our [📕 docs page](https://docs.skyvern.com).

## Supported LLMs
*(See original README for complete list)*

## Feature Roadmap
*(See original README)*

## Contributing

We welcome contributions!  See our [contribution guide](CONTRIBUTING.md) and  ["Help Wanted" issues](https://github.com/skyvern-ai/skyvern/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Telemetry

By Default, Skyvern collects basic usage statistics. To opt-out, set the `SKYVERN_TELEMETRY` environment variable to `false`.

## License
Skyvern is licensed under the [AGPL-3.0 License](LICENSE), with the exception of anti-bot measures available in our managed cloud offering.