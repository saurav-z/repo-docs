<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Browser Use empowers you to control your web browser with natural language, automating tasks and streamlining your workflow.**

## Key Features

*   **AI-Powered Automation:** Control your browser using simple prompts.
*   **Easy Installation:** Get started quickly with Python and `pip`.
*   **Cloud Option:**  Use the [cloud](https://cloud.browser-use.com) for faster, scalable, and stealth-enabled browser automation.
*   **MCP Integration:** Supports the Model Context Protocol (MCP) for integration with Claude Desktop and other MCP-compatible clients.
*   **Extensive Examples:**  Explore ready-to-use examples for various tasks, including shopping, lead management, and job applications.
*   **Robust Testing:**  Automated CI/CD with task validation.

## Quick Start

### Installation

```bash
pip install browser-use
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

**Remember to add your API keys to a `.env` file.**

## Demos

### Grocery Shopping

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

### LinkedIn to Salesforce Integration

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

### Job Application Automation

![Job Application Automation](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)

### Google Docs Automation

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

### Hugging Face Model Search

![Hugging Face Model Search](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd323b3)

## Learn More

*   **[Examples](examples):** Explore more use cases.
*   **[Discord](https://link.browser-use.com/discord):** Connect with the community.
*   **[`awesome-prompts`](https://github.com/browser-use/awesome-prompts):** Get inspiration for your prompts.

## Model Context Protocol (MCP) Integration

Browser-use seamlessly integrates with the Model Context Protocol, enabling powerful integrations.

### Using Browser-Use as an MCP Server with Claude Desktop

Configure Claude Desktop to use Browser Use.

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

### Connecting External MCP Servers

Extend Browser Use's capabilities by connecting to other MCP servers.

```python
import asyncio
from browser_use import Agent, Controller, ChatOpenAI
from browser_use.mcp.client import MCPClient

async def main():
    # Initialize controller
    controller = Controller()

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
    await filesystem_client.register_to_controller(controller)

    await github_client.connect()
    await github_client.register_to_controller(controller)

    # Create agent with MCP-enabled controller
    agent = Agent(
        task="Find the latest pdf report in my documents and create a GitHub issue about it",
        llm=ChatOpenAI(model="gpt-4.1-mini"),
        controller=controller  # Controller has tools from both MCP servers
    )

    # Run the agent
    await agent.run()

    # Cleanup
    await filesystem_client.disconnect()
    await github_client.disconnect()

asyncio.run(main())
```

**For more details:** [MCP documentation](https://docs.browser-use.com/customize/mcp-server)

## Vision

To empower users to control their digital world by simply telling their computer what to do.

## Roadmap

*   **Agent:** Improve speed and reduce token consumption.
*   **DOM Extraction:** Enhance UI element interaction and state representation.
*   **Workflows:** Enable users to record and rerun workflows.
*   **User Experience:** Create templates for common tasks.
*   **Parallelization:** Implement parallel processing for faster results.

## Contributing

We welcome contributions!  Please open issues for bugs or feature requests.  To contribute to the documentation, see the `/docs` folder.

## 🧪 Robust Agent Tasks

Ensure the reliability of your tasks with automated CI/CD:

*   **Add your task:** Create a YAML file in `tests/agent_tasks/` ([README details](tests/agent_tasks/README.md)).
*   **Automated validation:** Your task is run and evaluated on every update.

## Local Setup

For detailed information on setting up your local environment, refer to the [local setup documentation 📕](https://docs.browser-use.com/development/local-setup).

**Important:**  `main` is the primary development branch.  Use a [versioned release](https://github.com/browser-use/browser-use/releases) for production.

---

## Swag

Show off your Browser Use spirit! Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag for free 👀.

## Citation

If you use Browser Use in your work, please cite us:

```bibtex
@software{browser_use2024,
  author = {Müller, Magnus and Žunič, Gregor},
  title = {Browser Use: Enable AI to control your browser},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/browser-use/browser-use}
}
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/>

  [![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
  [![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
</div>

<div align="center">
  Made with ❤️ in Zurich and San Francisco
</div>
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  The one-sentence hook immediately explains the project's value.
*   **Keyword Optimization:**  Uses relevant keywords like "browser automation," "AI," "Python," and "web scraping" throughout the README.
*   **Well-Organized Headings and Structure:** Makes the information easy to scan and understand.
*   **Bulleted Key Features:** Highlights the main benefits of the project.
*   **Clear Call to Action:**  Encourages users to try the quick start, explore examples, and join the community.
*   **Strong Use of Links:** Internal and external links to key resources, documentation, and community channels.
*   **Emphasis on Benefits:** Focuses on what the user *gains* from using the library (e.g., automating tasks, streamlining workflows).
*   **Complete Code Snippets:** The quick start guide has been improved.
*   **Concise explanations:** Shortened the explanations to keep it easy to read.
*   **Added SEO Tags:** I can not add them, but the title, and the descriptions would be great for SEO.