<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser-use: Unleash AI to Automate Your Browser 🤖</h1>

<p align="center">
  <a href="https://github.com/browser-use/browser-use" target="_blank">
    <img src="https://img.shields.io/github/stars/browser-use/browser-use?style=social" alt="GitHub stars">
  </a>
  <a href="https://discord.gg/Tj2eUeUjE3" target="_blank">
    <img src="https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="https://cloud.browser-use.com" target="_blank">
    <img src="https://img.shields.io/badge/Cloud-☁️-blue" alt="Cloud">
  </a>
  <a href="https://docs.browser-use.com" target="_blank">
    <img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Documentation">
  </a>
  <a href="https://x.com/gregpr07" target="_blank">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Twitter - Gregor">
  </a>
  <a href="https://x.com/mamagnus00" target="_blank">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Twitter - Magnus">
  </a>
  <a href="https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341" alt="Weave Badge">
  </a>
</p>

**Browser-use empowers your AI agents to interact with web browsers, automating tasks and unlocking new possibilities.**  Check out the original repository on [GitHub](https://github.com/browser-use/browser-use).

## Key Features

*   **Effortless Browser Automation:** Connect AI agents to your browser for seamless task execution.
*   **Cloud-Based Deployment:** Easily try it out with our hosted version. **[Try the cloud ☁︎](https://cloud.browser-use.com)**.
*   **Model Context Protocol (MCP) Integration:** Integrate with Claude Desktop and other MCP-compatible clients for advanced functionality.
*   **Interactive CLI:** Test and experiment with the browser-use CLI.
*   **Comprehensive Examples:** Get started quickly with ready-to-use examples.
*   **Community & Support:** Join the [Discord](https://link.browser-use.com/discord) community to connect, share, and learn.
*   **Robustness Testing:** Ensure reliability with automated task validation in our CI/CD pipelines.

## Quick Start

1.  **Install:**

    ```bash
    pip install browser-use
    ```

2.  **Install Browser:**

    ```bash
    playwright install chromium --with-deps --no-shell
    ```

3.  **Run your Agent:**

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

4.  **Configure API Keys:** Add your API keys (OpenAI, Anthropic, etc.) to your `.env` file.

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

5.  **Explore the Documentation:**  Find detailed information on settings, models, and more in the [documentation 📕](https://docs.browser-use.com).

## Testing

*   **Web UI:** Test browser-use using its [Web UI](https://github.com/browser-use/web-ui).
*   **Desktop App:** Test browser-use using its [Desktop App](https://github.com/browser-use/desktop).
*   **Interactive CLI:**

    ```bash
    pip install "browser-use[cli]"
    browser-use
    ```

## MCP Integration

Browser-use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling integration with Claude Desktop and other MCP-compatible clients.

### Use as MCP Server with Claude Desktop

Add browser-use to your Claude Desktop configuration:

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

Browser-use agents can connect to multiple external MCP servers to extend their capabilities:

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

See the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

## Demos

**(Images and links to the demos are preserved)**

## Examples
Explore the [examples](examples) folder for practical use cases, or get inspired in our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.  Share your projects in the [Discord](https://link.browser-use.com/discord)!

## Vision

*   Make your computer act on your intent.

## Roadmap

**(Roadmap items are preserved)**

## Contributing

We welcome contributions!  Please open issues for bugs or feature requests.  Contribute to the docs in the `/docs` folder.

## 🧪 Robust Agent Testing

Run your tasks in our CI to ensure every update is working:

*   **Add a Task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic Validation:**  Your task is automatically run and evaluated on every update.

## Local Setup

For local setup instructions, consult the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:**  `main` is the development branch; for stable production, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com).  Good contributors receive swag for free 👀.

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