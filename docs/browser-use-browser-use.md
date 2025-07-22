<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Browser Use: Automate Your Browser with AI 🤖

**Effortlessly connect your AI agents to the web and automate tasks like never before.**  [Explore the original repository](https://github.com/browser-use/browser-use) for more details.

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Automation:**  Control your browser using the power of AI.
*   **Easy Integration:**  Simple Python setup for connecting agents to the browser.
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients.
*   **Hosted Version:** Try our [cloud version](https://cloud.browser-use.com) for instant browser automation.
*   **Extensive Documentation:**  Comprehensive documentation available at [docs.browser-use.com](https://docs.browser-use.com).
*   **Interactive CLI:** Use the browser-use interactive CLI for easy testing.
*   **Robust CI Testing:** Contribute and get automatic validation through CI.

## Quick Start

1.  **Install:**

    ```bash
    pip install browser-use
    ```
2.  **Install Browser:**

    ```bash
    playwright install chromium --with-deps --no-shell
    ```
3.  **Spin up your agent:**

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
4.  **Configure API Keys:** Add your API keys to a `.env` file.

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

## Examples

Browser Use makes it easy to automate a wide variety of tasks:

*   **Shopping:**  Add grocery items to cart and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **Lead Generation:**  Add your latest LinkedIn follower to your leads in Salesforce.
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application:**  Read your CV & find ML jobs, save them to a file, and then start applying.
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

*   **Document Generation:**  Write a letter in Google Docs and save it as a PDF.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Data Extraction & Processing:**  Look up models on Hugging Face, sort by likes, and save the top 5 to a file.
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

    Find more examples in the [examples](examples) folder or in our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## MCP Integration

Integrate with MCP-compatible tools like Claude Desktop for extended functionality:

*   **Use as MCP Server:** Configure Browser Use within Claude Desktop:

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

*   **Connect External MCP Servers:**  Extend agent capabilities with multiple servers:

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

    See the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for details.

## Roadmap

*   **Agent Enhancements:**  Improve memory and planning capabilities, and reduce token consumption.
*   **DOM Extraction:**  Enhance UI element detection and representation.
*   **Workflows:**  Implement workflow recording and rerunning.
*   **User Experience:**  Develop templates and improve documentation.
*   **Parallelization:** Enable parallel task execution for increased efficiency.

## Contributing

We welcome contributions!  Report bugs and request features by opening issues. Contribute to the docs in the `/docs` folder.

## 🧪 Robust Agent Testing

Ensure the reliability of your agents:

*   **Add your task:**  Create a YAML file in `tests/agent_tasks/` (see the [`README`](tests/agent_tasks/README.md) for details).
*   **Automated Validation:**  Your task is automatically run and evaluated on every update via CI.

## Local Setup

To learn more about the library, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:** The `main` branch is for active development. For production use, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Want to show off your Browser-use swag? Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag for free 👀.

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