<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI</h1>

**Browser Use empowers you to control your browser with the power of AI, enabling automation and intelligent web interaction.**  Check out the original repository [here](https://github.com/browser-use/browser-use).

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Browser Control:** Easily connect your AI agents to interact with web pages.
*   **Cloud Integration:**  Try the [hosted version](https://cloud.browser-use.com) for instant browser automation.
*   **Model Context Protocol (MCP) Support:** Integrates with MCP-compatible clients like Claude Desktop for extended capabilities.
*   **Interactive CLI:** Use the built-in CLI for quick testing and experimentation.
*   **Extensive Documentation:**  Comprehensive documentation available at [docs.browser-use.com](https://docs.browser-use.com).
*   **Community & Support:** Join the community in our [Discord](https://link.browser-use.com/discord) to share projects and get help.

## Quick Start

1.  **Installation:**

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

4.  **Configure API Keys:** Add your API keys to your `.env` file.

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

## MCP Integration

Integrate Browser Use with MCP-compatible applications such as Claude Desktop.

### Use as MCP Server with Claude Desktop

Configure Claude Desktop to use Browser Use:

```json
{
  "mcpServers": {
    "browser-use": {
      "command": "uvx",
      "args": ["browser-use", "--mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Connect External MCP Servers to Browser-Use Agent

```python
import asyncio
from browser_use import Agent, Controller
from browser_use.mcp.client import MCPClient
from browser_use.llm import ChatOpenAI

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
        task="Find the latest report.pdf in my documents and create a GitHub issue about it",
        llm=ChatOpenAI(model="gpt-4o"),
        controller=controller  # Controller has tools from both MCP servers
    )
    
    # Run the agent
    await agent.run()
    
    # Cleanup
    await filesystem_client.disconnect()
    await github_client.disconnect()

asyncio.run(main())
```

## Demos

Explore the capabilities of Browser Use with these examples:

*   [Shopping Example](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py)
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   LinkedIn to Salesforce Integration
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   [Job Application Example](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py)
*   [Letter Writing Example](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py)
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   [Hugging Face Model Search Example](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py)

## Vision

Enabling AI-driven web interaction.

## Roadmap

*   **Agent:** Improve agent memory, planning, and reduce token consumption.
*   **DOM Extraction:** Enhance UI element detection and representation.
*   **Workflows:** Implement workflow recording and rerunning.
*   **User Experience:**  Create templates, improve documentation, and increase speed.
*   **Parallelization:** Implement parallel processing for efficiency.

## Contributing

Contributions are welcome!  Please open issues for bugs or feature requests. For contributing to documentation, check the `/docs` folder.

## 🧪 Robust Agent Testing

We offer automated CI testing for your tasks.
*   **Add Your Task:** Add a YAML file in `tests/agent_tasks/` (see [`README`](tests/agent_tasks/README.md)).
*   **Automatic Validation:** Your task will be run and evaluated on every update.

## Local Setup

Learn more about local setup in the [documentation 📕](https://docs.browser-use.com/development/local-setup).

Install stable [versioned releases](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Get Browser-use swag! Check out our [Merch store](https://browsermerch.com). Great contributors get swag for free 👀.

## Citation

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