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

**Browser Use empowers you to control your web browser using natural language, allowing for unprecedented automation and efficiency.**

## Key Features

*   **AI-Powered Automation:** Control your browser using simple, natural language prompts.
*   **Cloud Integration:** Utilize the [Browser Use Cloud](https://cloud.browser-use.com) for fast, scalable, and stealth-enabled browser automation, skipping the setup.
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients for enhanced capabilities.
*   **Extensive Examples:** Explore a variety of [demos](#demos) and use cases to get started.
*   **Robust Testing:** Ensure your agents' reliability with automated CI task validation.
*   **Community & Support:** Join the [Discord](https://link.browser-use.com/discord) for help and show off your projects.
*   **Open Source:** Contribute to the project and shape the future of browser automation.
*   **MCP Integration:** Integrate with Claude Desktop and other MCP-compatible clients.

## Quick Start

Install Browser Use using pip:

```bash
pip install browser-use
```

**(Optional) Install Chromium using Playwright:**

```bash
uvx playwright install chromium --with-deps --no-shell
```

**Spin up your agent:**

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

**Add your API keys to your .env file:**

```bash
OPENAI_API_KEY=
```

For detailed setup and configuration, explore the [documentation](https://docs.browser-use.com).

## Demos

Explore these example use cases to see Browser Use in action:

*   **Shopping Automation:** Automate grocery shopping (see video)
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce Integration:** Automatically add leads.
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   **Job Application Automation:** Find and apply for jobs based on your CV.
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
*   **Document Creation:** Generate documents in Google Docs.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **File Management:** Search and download models from Hugging Face.
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

For even more examples, explore the [examples](examples) folder and the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repository.

## MCP Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), extending its capabilities:

### Use as MCP Server with Claude Desktop

Integrate with Claude Desktop by configuring it to use Browser Use:

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

Extend Browser Use agents with external MCP servers:

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

Refer to the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for further details.

## Vision

Our vision is to empower users to control their computers through natural language.

## Roadmap

We are continuously improving Browser Use. Here's a glimpse of our ongoing development:

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

We welcome contributions! Please feel free to open issues for bugs or feature requests.  For documentation contributions, check out the `/docs` folder.

## 🧪 Robust Agent Testing

Ensure your agents' reliability:

*   **Create Task:** Add a YAML file in `tests/agent_tasks/` (see [`README`](tests/agent_tasks/README.md) for details).
*   **Automated Validation:** Your task will be automatically tested on every update.

## Local Setup

Learn more about the project with the [local setup guide](https://docs.browser-use.com/development/local-setup).

**Important:** `main` is the development branch. For stable use, install a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser Use pride! Check out the [Merch store](https://browsermerch.com). Good contributors receive free swag. 👀

## Citation

If you use Browser Use in your research, please cite us:

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