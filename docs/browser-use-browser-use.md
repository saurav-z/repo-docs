<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Unleash AI to Automate Your Browser</h1>

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

✨ **Browser Use empowers you to control your browser with natural language, automating tasks and unlocking new possibilities.**

🌤️ **Skip the setup and get started instantly with our [cloud](https://cloud.browser-use.com) solution!**

## Key Features

*   **AI-Powered Automation:** Control your browser using natural language.
*   **Easy Setup:**  Quickly integrate with Python.
*   **Versatile Use Cases:** Automate web scraping, form filling, and more.
*   **Integration with Model Context Protocol (MCP):** Connect with Claude Desktop and other MCP-compatible clients for advanced functionality.
*   **Cloud Option:** Get faster, scalable, and stealth-enabled browser automation.

## 🎉 OSS Twitter Hackathon

We just hit **69,000 GitHub ⭐**!
To celebrate, we're launching **#nicehack69** — a Twitter-first hackathon with a **$6,900 prize pool**. Dream big and show us the future of browser-use agents that go beyond demos!

**Deadline: September 10, 2025**

**[🚀 Join the hackathon →](https://github.com/browser-use/nicehack69)**

<div align="center">
<a href="https://github.com/browser-use/nicehack69">
<img src="./static/NiceHack69.png" alt="NiceHack69 Hackathon" width="600"/>
</a>
</div>

> **🚀 Use the latest version!** 
> 
> We ship every day improvements for **speed**, **accuracy**, and **UX**. 
> ```bash
> pip install --upgrade browser-use
> ```

## Quickstart

1.  **Installation:**
    ```bash
    pip install browser-use
    ```
    If you don't already have Chrome or Chromium installed, you can also download the latest Chromium using playwright's install shortcut:

    ```bash
    uvx playwright install chromium --with-deps --no-shell
    ```
2.  **Get started:**
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
3.  **Configure:** Add your API keys to your `.env` file.
    ```bash
    OPENAI_API_KEY=
    ```
4.  **Documentation:** Explore further options and customization in the [documentation 📕](https://docs.browser-use.com).

## Examples & Use Cases

*   **Shopping Automation:** Add items to cart and checkout.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **Lead Generation:** Add LinkedIn followers to Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application:** Find and apply for jobs based on your CV.

    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

*   **Document Generation:** Write a letter in Google Docs and save as PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Data Extraction:** Find models with specific licenses and save the top results.

    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

### More Examples

Find more examples in the [examples](examples) folder or visit the [Discord](https://link.browser-use.com/discord) to showcase your projects. For inspiration, see our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## Model Context Protocol (MCP) Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling easy integration with Claude Desktop and other compatible clients.

### Using as MCP Server with Claude Desktop

Integrate Browser Use with Claude Desktop:

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

### Connecting External MCP Servers to Browser-Use Agent

Extend your agent's capabilities:

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

Explore the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for detailed information.

## Vision

Envision a future where you simply tell your computer what to do, and it's done.

## Roadmap

### Agent

-   [ ] Make agent 3x faster
-   [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

-   [ ] Enable interaction with all UI elements
-   [ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows

-   [ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience

-   [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization

-   [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions! Report bugs and request features by opening issues. For documentation contributions, see the `/docs` folder.

## 🧪 How to make your agents robust?

We offer to run your tasks in our CI—automatically, on every update!

-   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
-   **Automatic validation:** Every time we push updates, your task will be run by the agent and evaluated using your criteria.

## Local Setup

To learn more about the library, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

`main` is the primary development branch with frequent changes. For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases) instead.

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag for free 👀.

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
```
Key improvements and optimizations:

*   **Stronger Headline:**  Used a more compelling and keyword-rich headline.
*   **Concise Hook:**  Created a one-sentence hook to grab attention.
*   **SEO-Friendly Keywords:** Incorporated keywords like "AI automation," "browser automation," and relevant task examples.
*   **Clear Structure:**  Organized content with clear headings and bullet points for readability.
*   **Action-Oriented Language:** Used phrases like "Unleash," "Get Started," and "Explore" to encourage engagement.
*   **Prioritized Information:** Placed key features and quickstart instructions at the top for immediate value.
*   **Call to Action:** Encouraged users to join the hackathon.
*   **Expanded Summaries:** Included more descriptive summaries for examples.
*   **MCP Explanation:** More clearly explained the value proposition of MCP integration.
*   **Roadmap Integration:** Added the roadmap to the readme.
*   **Concise Vision Section**
*   **Improved Formatting:** Ensured consistent markdown formatting.

This revised README is more user-friendly, highlights the core benefits effectively, and is optimized for search engines, ultimately increasing the visibility and appeal of the project.