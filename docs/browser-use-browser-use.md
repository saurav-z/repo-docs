<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo - Dark and Light Modes" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

**Control your web browser effortlessly using AI!**  [Explore the original repository on GitHub](https://github.com/browser-use/browser-use).

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow Gregor](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow Magnus](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language prompts.
*   **Cloud-Based Option:**  Use the [cloud](https://cloud.browser-use.com) for faster, scalable, and stealth-enabled automation.
*   **Model Context Protocol (MCP) Integration:** Compatible with Claude Desktop and other MCP clients.
*   **Extensive Examples:**  Get started quickly with ready-to-use demos.
*   **Open Source:**  Contribute to the project!

## Quick Start

Install using pip:

```bash
pip install browser-use
```

Install Chromium (if you don't have it) with Playwright:

```bash
uvx playwright install chromium --with-deps --no-shell
```

Example Python code:

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

Add your API keys (e.g., `OPENAI_API_KEY=`) to a `.env` file.  See the [documentation 📕](https://docs.browser-use.com) for detailed setup and configuration.

## Demos & Examples

Explore these examples of what you can achieve with Browser Use:

*   **Shopping:**  Add items to a cart and checkout.  [Watch the demo](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce:** Automate lead generation.  [View example](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   **Job Application:**  Find and apply for jobs.  [View example](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)
*   **Google Docs:**  Generate and save documents.  [View example](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **Hugging Face:**  Search and save models.  [View example](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

Find more examples in the [`examples`](examples) folder or get inspired in the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## MCP Integration

Browser Use seamlessly integrates with the Model Context Protocol (MCP).

### Use as MCP Server with Claude Desktop

Configure Claude Desktop to utilize Browser Use:

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

Extend your agents' functionality:

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

More details in the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Vision

Empowering you to control your computer with natural language.

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

Contributions are welcome!  Report bugs and request features by opening issues.  Contribute to the docs by editing the `/docs` folder.

## 🧪 Robust Agent Testing

Enhance your agents with automated testing in our CI.

*   **Add your task:** Create a YAML file in `tests/agent_tasks/` (see [`README there`](tests/agent_tasks/README.md)).
*   **Automated validation:** Your tasks are run on every update and evaluated based on your criteria.

## Local Setup

For local setup and more library information, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

Use a stable [versioned release](https://github.com/browser-use/browser-use/releases) for production.

---

## Swag

Show off your Browser Use pride! Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag 👀.

## Citation

If you use Browser Use in your research, please cite:

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
 
[![Twitter Follow Gregor](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow Magnus](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
 
 </div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>