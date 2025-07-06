<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Effortlessly control your browser using the power of AI with Browser Use, enabling automation and complex web interactions.** ([Back to Original Repo](https://github.com/browser-use/browser-use))

## Key Features

*   **AI-Powered Browser Automation:**  Enable your AI agents to interact with web browsers, performing tasks automatically.
*   **Easy Integration:** Simple installation with pip, making it easy to get started.
*   **Versatile LLM Support:**  Integrate with a variety of LLMs, including OpenAI, Anthropic, and more.
*   **Cloud-Based Option:**  Try the hosted version for instant browser automation without setup.
*   **Interactive CLI:** Test and experiment with the `browser-use` CLI.
*   **Robust Testing:** Integrated testing with CI for automated validation of your tasks.

## Quick Start

### Installation

Install the package and browser dependencies:

```bash
pip install browser-use
playwright install chromium --with-deps --no-shell
```

### Example Usage

Here's a simple example to get you started:

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from browser_use.llm import ChatOpenAI

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="o4-mini", temperature=1.0),
    )
    await agent.run()

asyncio.run(main())
```

### Configuration

Configure your API keys using a `.env` file:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
GOOGLE_API_KEY=
DEEPSEEK_API_KEY=
GROK_API_KEY=
NOVITA_API_KEY=
```

### Additional Resources

*   **Documentation:** Find more details and advanced configurations in the [documentation 📕](https://docs.browser-use.com).
*   **Web UI & Desktop App:** Test browser-use using the [Web UI](https://github.com/browser-use/web-ui) or [Desktop App](https://github.com/browser-use/desktop).
*   **Interactive CLI:** Use our `browser-use` interactive CLI:  `pip install "browser-use[cli]" && browser-use`

## Demos

See Browser Use in action:

*   **Grocery Shopping:**  Automated grocery shopping [Demo Video](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce:**  Automate lead generation.
*   **Job Application:** Find and apply for jobs based on your CV.
*   **Document Creation:** Generate a letter in Google Docs and save it as a PDF.
*   **Hugging Face Model Search:** Search and save models from Hugging Face.

(Embed the demo images with alt text, see original for examples)

## More Examples

Explore more use cases and get inspiration:

*   [Examples Folder](examples)
*   [Discord](https://link.browser-use.com/discord)
*   [`awesome-prompts`](https://github.com/browser-use/awesome-prompts)

## Vision

The future of web interaction is here: tell your computer what to do, and it gets it done.

## Roadmap

### Agent
-   [ ] Improve agent memory to handle +100 steps
-   [ ] Enhance planning capabilities (load website specific context)
-   [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction
-   [ ] Enable detection for all possible UI elements
-   [ ] Improve state representation for UI elements so that all LLMs can understand what's on the page

### Workflows
-   [ ] Let user record a workflow - which we can rerun with browser-use as a fallback
-   [ ] Make rerunning of workflows work, even if pages change

### User Experience
-   [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
-   [ ] Improve docs
-   [ ] Make it faster

### Parallelization
-   [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions!

*   Report bugs and request features by opening issues.
*   Contribute to the documentation in the `/docs` folder.

## 🧪 Robust Agent Testing

Automate the validation of your tasks:

*   Create a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   Your task is run on every update and evaluated against your criteria.

## Local Setup

For more details, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

## Versioning

`main` is the primary development branch.  For production, install a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Good contributors receive swag! Check out our [Merch store](https://browsermerch.com) to show off your Browser Use love.

## Citation

If you use Browser Use in your research or project, please cite:

```bibtex
@software{browser_use2024,
  author = {Müller, Magnus and Žunič, Gregor},
  title = {Browser Use: Enable AI to control your browser},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/browser-use/browser-use}
}
```

 <div align="center"> <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/> 
 
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
 
 </div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>