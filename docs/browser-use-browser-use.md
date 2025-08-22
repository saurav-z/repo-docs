<!-- Improved README for Browser Use -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

<div align="center">
  <a href="https://github.com/browser-use/browser-use">
    <img src="https://img.shields.io/github/stars/gregpr07/browser-use?style=social" alt="GitHub stars">
  </a>
  <a href="https://discord.gg/5zXwJ39vYn">
    <img src="https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="https://cloud.browser-use.com">
    <img src="https://img.shields.io/badge/Cloud-☁️-blue" alt="Cloud">
  </a>
  <a href="https://docs.browser-use.com">
    <img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Documentation">
  </a>
  <a href="https://x.com/intent/user?screen_name=gregpr07">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Twitter - Gregor">
  </a>
  <a href="https://x.com/intent/user?screen_name=mamagnus00">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Twitter - Magnus">
  </a>
  <a href="https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615">
      <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341" alt="WorkWeave Badge">
  </a>
</div>

<p align="center"><b>Empower your AI to interact with the web effortlessly using Browser Use, the leading browser automation tool.</b></p>

## Key Features

*   **AI-Driven Automation:** Control your browser with natural language prompts.
*   **Cloud Integration:** Use our cloud service for faster setup and scalability.
*   **Model Context Protocol (MCP) Integration:** Connect with Claude Desktop and other MCP-compatible clients.
*   **Robust Example Demos:** Browse various use cases like shopping, and job applications, from start to finish.
*   **Customizable & Extensible:** Integrate with other tools and extend functionality.
*   **Thorough Documentation:** Clear, comprehensive documentation to guide you through the process.

## Quick Start

1.  **Installation (with Python>=3.11):**

    ```bash
    pip install browser-use
    ```

2.  **Install Chrome/Chromium (if needed):**

    ```bash
    uvx playwright install chromium --with-deps --no-shell
    ```

3.  **Example Usage:**

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

4.  **API Keys:** Add your API keys to your `.env` file:

    ```bash
    OPENAI_API_KEY=YOUR_API_KEY
    ```

5.  **Documentation:** Explore the full documentation for more details and advanced configuration options: [Documentation 📕](https://docs.browser-use.com).

## Demos & Examples

Explore various use cases:

*   **Shopping:** Automate adding items to a cart and checking out.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce Integration:** Automatically add new LinkedIn followers to your Salesforce leads.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application Automation:** Find and apply for ML jobs.
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

*   **Document Generation:** Create and save documents in Google Docs as PDFs.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Hugging Face Model Search:** Find and save top models based on specific criteria.
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

*   **More Examples:**  Explore additional use cases in the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) to share your projects.  Check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting inspiration.

## MCP Integration

Browser-use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling integration with Claude Desktop and other MCP-compatible clients.

### Use as MCP Server with Claude Desktop

Add browser-use to your Claude Desktop configuration:

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

This gives Claude Desktop access to browser automation tools for web scraping, form filling, and more.

### Connect External MCP Servers to Browser-Use Agent

Browser-use agents can connect to multiple external MCP servers to extend their capabilities:

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

See the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

## Vision

Browser Use is designed to make your AI agent act and function as a human browser user.

## Roadmap

*   **Agent:**
    *   Make agent 3x faster
    *   Reduce token consumption (system prompt, DOM state)
*   **DOM Extraction:**
    *   Enable interaction with all UI elements
    *   Improve state representation for UI elements so that any LLM can understand what's on the page
*   **Workflows:**
    *   Let user record a workflow - which we can rerun with browser-use as a fallback
*   **User Experience:**
    *   Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
*   **Parallelization:**
    *   Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

Contributions are welcome!  Please open issues for bugs, feature requests, or contribute to the documentation in the `/docs` folder.

## 🧪 Agent Task Validation

We run your tasks automatically in our CI.

*   **Add your task:** Create a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Your task runs and is evaluated based on your criteria.

## Local Setup

Learn more about local setup by visiting the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:** `main` is the development branch.  Use a [versioned release](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Show off your Browser Use swag!  Visit our [Merch store](https://browsermerch.com).  Good contributors can receive free swag 👀.

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