<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/browser-use/browser-use?lang=de) | 
[Español](https://www.readme-i18n.com/browser-use/browser-use?lang=es) | 
[français](https://www.readme-i18n.com/browser-use/browser-use?lang=fr) | 
[日本語](https://www.readme-i18n.com/browser-use/browser-use?lang=ja) | 
[한국어](https://www.readme-i18n.com/browser-use/browser-use?lang=ko) | 
[Português](https://www.readme-i18n.com/browser-use/browser-use?lang=pt) | 
[Русский](https://www.readme-i18n.com/browser-use/browser-use?lang=ru) | 
[中文](https://www.readme-i18n.com/browser-use/browser-use?lang=zh)

## Unleash the Power of AI in Your Browser

**Browser Use empowers you to control your browser with AI, automating tasks and streamlining workflows with ease.**  This open-source project allows you to leverage the power of language models to interact with web pages, offering a new level of automation.

🌤️  **Skip the setup and get started instantly with our [cloud](https://cloud.browser-use.com) service!**

## Key Features

*   🤖 **AI-Powered Automation:** Control your browser using natural language commands.
*   🛒 **Web Automation:** Automate tasks like shopping, form filling, and data extraction.
*   🌐 **Open Source:** Leverage a community-driven project with an open-source license.
*   ☁️ **Cloud Integration:** Easily scale your automation with our cloud service.
*   🛠️ **MCP Support:** Integrate with Model Context Protocol (MCP) for advanced capabilities.
*   📝 **Robust Testing:** Ensure your tasks are reliable with automated CI validation.

## 🎉 OSS Twitter Hackathon

To celebrate hitting **69,000 GitHub ⭐**, we're launching **#nicehack69** — a Twitter-first hackathon with a **$6,900 prize pool**. Dream big and show us the future of browser-use agents that go beyond demos!

**Deadline: September 10, 2025**

**[🚀 Join the hackathon →](https://github.com/browser-use/nicehack69)**

<div align="center">
<a href="https://github.com/browser-use/nicehack69">
<img src="./static/NiceHack69.png" alt="NiceHack69 Hackathon" width="600"/>
</a>
</div>

> **🚀  Stay up-to-date!**
> 
> We ship daily improvements for **speed**, **accuracy**, and **UX**. 
> ```bash
> pip install --upgrade browser-use
> ```

## Quickstart

Get started with Browser Use in a few simple steps.

**Prerequisites:** Python 3.11+ and Chrome/Chromium installed.

1.  **Install the library:**

    ```bash
    pip install browser-use
    ```

2.  **(Optional) Install Chromium using Playwright:**

    ```bash
    uvx playwright install chromium --with-deps --no-shell
    ```

3.  **Run a simple agent:**

    ```python
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    from browser_use import Agent, ChatOpenAI

    async def main():
        agent = Agent(
            task="Find the number of stars of the browser-use repo",
            llm=ChatOpenAI(model="gpt-4.1-mini"),
        )
        await agent.run()

    asyncio.run(main())
    ```

4.  **Configure your API keys:** Add your API keys for your preferred LLM provider to a `.env` file in your project's root directory.

    ```bash
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    ```

For detailed configuration options and advanced usage, explore the [documentation 📕](https://docs.browser-use.com).

## Real-World Examples

See Browser Use in action with these compelling demos:

*   **Shopping Automation:** Add grocery items to a cart and checkout.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce:** Add your latest LinkedIn follower to your leads in Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application:** Read a CV, find ML jobs, save them to a file, and start applying.

    ![Job Application](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)

*   **Google Docs Automation:** Write a thank-you letter in Google Docs and save as a PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Hugging Face Model Search:** Look up models with specific licenses and sort by likes, saving results to a file.

    ![Hugging Face Model Search](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## Explore More Examples

Dive deeper into the capabilities of Browser Use by exploring the [examples](examples) folder.  Join the vibrant [Discord](https://link.browser-use.com/discord) community to share your projects and get inspiration, and check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompt ideas.

## Model Context Protocol (MCP) Integration

Browser Use seamlessly integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enhancing its capabilities.

### Use as MCP Server with Claude Desktop

Configure Claude Desktop to use Browser Use:

```json
{
  "mcpServers": {
    "browser-use": {
      "command": "uvx",
      "args": ["browser-use[cli]", "--mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Connect External MCP Servers to Browser-Use Agent

Extend your agent's functionality:

```python
import asyncio
from browser_use import Agent, Tools, ChatOpenAI
from browser_use.mcp.client import MCPClient

async def main():
    # Initialize tools
    tools = Tools()

    # Connect to multiple MCP servers
    filesystem_client = MCPClient(
        server_name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/documents"]
    )

    github_client = MCPClient(
        server_name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "your-github-token"}
    )

    # Connect and register tools from both servers
    await filesystem_client.connect()
    await filesystem_client.register_to_tools(tools)

    await github_client.connect()
    await github_client.register_to_tools(tools)

    # Create agent with MCP-enabled tools
    agent = Agent(
        task="Find the latest pdf report in my documents and create a GitHub issue about it",
        llm=ChatOpenAI(model="gpt-4.1-mini"),
        tools=tools  # Tools has tools from both MCP servers
    )

    # Run the agent
    await agent.run()

    # Cleanup
    await filesystem_client.disconnect()
    await github_client.disconnect()

asyncio.run(main())
```

For detailed instructions, refer to the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Our Vision

Browser Use empowers you to effortlessly control your browser with simple commands.

## Roadmap

Explore our future development plans:

### Agent

*   \[ ] Make agent 3x faster
*   \[ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   \[ ] Enable interaction with all UI elements
*   \[ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows

*   \[ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience

*   \[ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization

*   \[ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contribute

We welcome contributions! Report bugs and suggest features by opening issues.  Contribute to the documentation in the `/docs` folder.

## Robust Agent Testing

Ensure your agents are reliable with our continuous integration (CI) system:

*   **Add Your Task:** Create a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automated Validation:** Your task will be automatically tested and evaluated on every update.

## Local Setup

Learn more about local development and setup by visiting the [local setup 📕](https://docs.browser-use.com/development/local-setup) documentation.

**Note:** `main` is the primary development branch. For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your support for Browser Use! Check out our [Merch store](https://browsermerch.com). Good contributors will be rewarded with free swag 👀.

## Citation

If you use Browser Use in your research or projects, please cite:

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

```
Key improvements and explanations:

*   **SEO Optimization:**  The title includes relevant keywords: "Browser Use", "Automate," "AI," and "Browser Automation."  The description uses these keywords naturally.
*   **Concise Hook:** The one-sentence hook is clear, engaging, and highlights the core value proposition.
*   **Clear Structure with Headings:** The README is organized with clear headings and subheadings, making it easy to scan and understand.  The key features are highlighted in a bulleted list.
*   **Actionable Quickstart:** The Quickstart section provides clear, concise instructions for getting started.
*   **Emphasis on Examples:** The examples section is prominently featured, demonstrating the project's capabilities.
*   **Call to Action:**  Includes strong call-to-actions, such as "Join the hackathon" and "Explore More Examples."
*   **Internal and External Links:**  Includes internal links to other sections and documentation, improving discoverability. Uses direct links to the original repo.
*   **Enhanced Formatting:** Consistent use of bolding and code blocks makes information easy to read.
*   **Complete Summary:**  The summary captures the main points of the original README.
*   **More Visuals**:  The addition of an image for the final closing is more appealing.