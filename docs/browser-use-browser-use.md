<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Browser Use: Unleash the Power of AI in Your Browser

**Effortlessly automate your browser with AI, turning complex tasks into simple prompts!**  ([Back to Top](#browser-use-unleash-the-power-of-ai-in-your-browser))

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language prompts.
*   **Easy Integration:** Simple installation and setup.
*   **Hosted Cloud Version:**  Get started instantly with our [hosted version](https://cloud.browser-use.com).
*   **Model Context Protocol (MCP) Support:** Integrates with Claude Desktop and other MCP-compatible clients.
*   **Extensible:** Connect with external MCP servers for enhanced capabilities.
*   **Robust Testing:**  Automated CI testing for reliable agent performance.
*   **Community & Resources:**  Join our [Discord](https://link.browser-use.com/discord) and explore our [awesome-prompts](https://github.com/browser-use/awesome-prompts) for inspiration.

## Quick Start

1.  **Installation (with Python>=3.11):**

    ```bash
    pip install browser-use
    ```

2.  **Install Browser Dependencies:**

    ```bash
    playwright install chromium --with-deps --no-shell
    ```

3.  **Example Usage:**

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

4.  **Configure API Keys:** Add your API keys to your `.env` file:

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

5.  **Documentation:**  For more detailed information, settings, and models, refer to the comprehensive [documentation](https://docs.browser-use.com).

## Test & Integrate

*   **Web UI:** Test with the [Web UI](https://github.com/browser-use/web-ui).
*   **Desktop App:** Test with the [Desktop App](https://github.com/browser-use/desktop).
*   **Interactive CLI:** Use the interactive CLI (similar to `claude`):

    ```bash
    pip install "browser-use[cli]"
    browser-use
    ```

## Model Context Protocol (MCP) Integration

Browser-use fully supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for advanced integration.

### Use as MCP Server with Claude Desktop

Integrate Browser Use with Claude Desktop. See instructions in the original README.

### Connect External MCP Servers to Browser-Use Agent

Connect Browser-use agents to multiple external MCP servers. See example code in the original README.

## Demos

Explore the power of Browser Use through these demos, showcasing diverse automation capabilities.

*   **Grocery Shopping:** [AI Did My Groceries](https://www.youtube.com/watch?v=L2Ya9PYNns8)
    *   Add grocery items to cart and checkout.
*   **LinkedIn to Salesforce:**  Prompt: Add my latest LinkedIn follower to my leads in Salesforce.
    *   <img src="https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07" alt="LinkedIn to Salesforce" width="400">
*   **Find and Apply for Jobs:**
    *   Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.
    *   <img src="https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04" alt="Find and Apply for Jobs" width="400">
*   **Create a Letter in Google Docs:**
    *   Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.
    *   <img src="https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa" alt="Letter to Papa" width="400">
*   **Hugging Face Model Search:**
    *   Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.
    *   <img src="https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3" alt="Hugging Face Model Search" width="400">

## More Examples

Browse the [examples](examples) folder for a variety of use cases and join our [Discord](https://link.browser-use.com/discord) to showcase your projects and get inspired. Also check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting inspiration.

## Vision

The ultimate goal is to enable users to simply tell their computer what to do and have it execute the instructions in the browser.

## Roadmap

The project is continuously evolving. Current focus areas include:

*   **Agent Improvements:** Enhancing memory, planning capabilities, and reducing token consumption.
*   **DOM Extraction:** Improving the detection and representation of UI elements.
*   **Workflows:** Enabling workflow recording and re-running.
*   **User Experience:** Creating templates, improving documentation, and increasing speed.
*   **Parallelization:** Implementing parallel task execution for efficiency.

## Contributing

We welcome contributions! Please feel free to open issues for bugs or feature requests.  Contribute to the docs by checking out the `/docs` folder.

## 🧪 Robustness & Testing

Make your agents robust with automated CI testing:

*   Add a YAML file in `tests/agent_tasks/` (see the [`README`](tests/agent_tasks/README.md) for details).
*   Your tasks will be automatically run and evaluated on every update.

## Local Setup

For detailed setup instructions, see the [local setup 📕](https://docs.browser-use.com/development/local-setup) documentation.

**Note:** `main` is the primary development branch.  For stable releases, install a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com). Good contributors may receive swag for free 👀.

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