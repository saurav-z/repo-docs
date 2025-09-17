<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Unleash AI to Automate Your Web Tasks</h1>

**Effortlessly automate your web browsing with the power of AI.**  [Explore the original repository](https://github.com/browser-use/browser-use).

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

<br>
🌤️ **Quick Start:** Skip the setup and use our [cloud](https://cloud.browser-use.com) for faster, scalable, and stealth-enabled browser automation!

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language.
*   **Easy to Use:** Simple installation and intuitive API.
*   **Versatile:** Automate a wide range of web tasks.
*   **Open Source:** Leverage the power of AI and browser automation.
*   **Extensible:** Integrates with Model Context Protocol (MCP).

**🚀 Use the latest version!** 

> We ship every day improvements for **speed**, **accuracy**, and **UX**. 
> ```bash
> uv pip install --upgrade browser-use
> ```

## Getting Started

### Installation

With `uv` (Python>=3.11):

```bash
uv pip install browser-use
```

If you don't already have Chrome or Chromium installed, you can also download the latest Chromium using playwright's install shortcut:

```bash
uvx playwright install chromium --with-deps --no-shell
```

### Basic Usage

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

*   **Configuration:** Add your API keys to your `.env` file.
    ```bash
    OPENAI_API_KEY=
    ```
*   **Documentation:** For detailed information, explore our [documentation 📕](https://docs.browser-use.com).

## Demos: See Browser Use in Action!

*   **Grocery Shopping:** Add items to cart and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce:**  Automate lead management.
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application Automation:** Find and apply for jobs.
    ![Find and Apply to Jobs](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)

*   **Document Generation:** Write a letter in Google Docs.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Hugging Face Model Search:** Find and save models.
    ![Hugging Face Models](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## More Examples & Community

*   **Explore:**  Find more examples in the [examples](examples) folder.
*   **Connect:** Join the [Discord](https://link.browser-use.com/discord) to showcase your projects.
*   **Inspiration:** Discover prompting techniques in our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repository.

## Model Context Protocol (MCP) Integration

Browser Use integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for enhanced capabilities.

### Use as MCP Server with Claude Desktop

Configure Claude Desktop to use browser-use:

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

Extend agent capabilities by connecting to multiple MCP servers:

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

Refer to the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

## Vision: The Future of Web Automation

Our goal: Make your computer do what you tell it to do, seamlessly.

## Roadmap

### Agent
*   [ ] Make agent 3x faster
*   [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction
*   [ ] Enable interaction with all UI elements
*   [ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows
*   [ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience
*   [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization
*   [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions! Please open issues for bugs or feature requests.  Contribute to the docs in the `/docs` folder.

## Robust Agent Testing

Automated CI testing available for your tasks:

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Your task will be tested on every update.

## Local Setup

See [local setup 📕](https://docs.browser-use.com/development/local-setup) for more information on how to develop the library.

## Versioning

`main` is the primary development branch. Use [versioned releases](https://github.com/browser-use/browser-use/releases) for stable production.

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