<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI</h1>

<div align="center">
  <a href="https://github.com/browser-use/browser-use">
    <img src="https://img.shields.io/github/stars/gregpr07/browser-use?style=social" alt="GitHub stars">
  </a>
  <a href="https://discord.gg/browser-use">
    <img src="https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="https://cloud.browser-use.com">
    <img src="https://img.shields.io/badge/Cloud-☁️-blue" alt="Cloud">
  </a>
  <a href="https://docs.browser-use.com">
    <img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Documentation">
  </a>
  <a href="https://x.com/gregpr07">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Follow Gregor on Twitter">
  </a>
  <a href="https://x.com/mamagnus00">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Follow Magnus on Twitter">
  </a>
  <a href="https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341" alt="WorkWeave Badge">
  </a>
</div>

**Browser Use empowers you to control your web browser using natural language, automating tasks and streamlining workflows.**

## Key Features

*   **AI-Powered Automation:** Automate browser tasks with simple, natural language prompts.
*   **Cloud Deployment:** Quickly get started with the [Browser Use Cloud](https://cloud.browser-use.com) for faster, scalable automation.
*   **Easy Installation:** Install with a simple `pip install browser-use` command.
*   **Flexible Integration:** Integrates with the Model Context Protocol (MCP) for enhanced capabilities.
*   **Extensive Examples:** Explore a variety of [examples](#demos) to get you started quickly.
*   **Robustness & Testing:**  Ensure your tasks work correctly with built-in CI testing capabilities.
*   **Community & Support:** Join the vibrant [Discord](https://link.browser-use.com/discord) community for help and to showcase your projects.

## Quick Start

Get up and running in minutes:

```bash
pip install browser-use
```

Install Chromium (if you don't have it already):

```bash
uvx playwright install chromium --with-deps --no-shell
```

Example usage:

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

Remember to add your API keys to your `.env` file:

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

For detailed information, check out the comprehensive [documentation](https://docs.browser-use.com).

## Demos

See Browser Use in action with these compelling examples:

*   **[AI Did My Groceries](https://www.youtube.com/watch?v=L2Ya9PYNns8):**  Automated grocery shopping.

    <br/>
    <img src="https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14" alt="AI Did My Groceries" width="400"/>
    <br/>

*   **LinkedIn to Salesforce:** Automate lead generation by adding your latest LinkedIn follower to Salesforce.

    <br/>
    <img src="https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07" alt="LinkedIn to Salesforce" width="400"/>
    <br/>

*   **Find & Apply for Jobs:** Leverage your CV to find and apply for ML jobs.

    <br/>
    <img src="https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04" alt="Find & Apply for Jobs" width="400"/>
    <br/>

*   **Letter to Papa:** Generate a thank-you letter to your Papa and save it as a PDF.

    <br/>
    <img src="https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa" alt="Letter to Papa" width="400"/>
    <br/>

*   **Find & Save Hugging Face Models:** Search Hugging Face for models and save the top results to a file.

    <br/>
    <img src="https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3" alt="Find & Save Hugging Face Models" width="400"/>
    <br/>

## More Resources

*   Explore more examples in the [examples](examples) folder.
*   Get inspired by the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.
*   Join the [Discord](https://link.browser-use.com/discord) to ask questions and share your projects.

## Model Context Protocol (MCP) Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing integration with tools like Claude Desktop.

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

### Connect External MCP Servers

Extend Browser Use's capabilities by connecting to other MCP servers:

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

Refer to the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for advanced usage.

## Vision

Our vision is to enable true AI-driven browser automation.

## Roadmap

*   **Agent Optimization:** Faster agent execution and reduced token consumption.
*   **Enhanced DOM Interaction:** Support for all UI elements and improved state representation.
*   **Workflow Recording:** Allow users to record and rerun workflows.
*   **Templates & Tutorials:** Create templates for common tasks (e.g., job applications, QA testing).
*   **Parallelization:** Enable parallel task execution for increased efficiency.

## Contributing

Contributions are welcome!  Please submit issues for bugs or feature requests.  Contribute to the docs in the `/docs` folder.

## 🧪 Automated Testing

Ensure your agents are reliable by using our automated testing framework:

*   Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   Your tasks will automatically be run and evaluated on every update.

## Local Setup

Learn more about setting up your development environment in the [local setup 📕](https://docs.browser-use.com/development/local-setup).

For production use, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser Use pride!  Check out our [Merch store](https://browsermerch.com).  Good contributors may receive free swag.

## Citation

If you use Browser Use in your research or project, please cite our work:

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

  <a href="https://x.com/gregpr07">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Follow Gregor on Twitter">
  </a>
  <a href="https://x.com/mamagnus00">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Follow Magnus on Twitter">
  </a>

</div>

<div align="center">
  Made with ❤️ in Zurich and San Francisco
</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  Includes relevant keywords like "AI," "browser automation," and "web automation."
*   **Concise Hook:**  The one-sentence introduction clearly states the core functionality and benefit.
*   **Clear Headings and Structure:** Improves readability and helps users quickly find information.
*   **Bulleted Key Features:** Highlights the main selling points in an easy-to-scan format.
*   **Emphasis on Benefits:**  Focuses on what the user *gets* from the tool (automation, efficiency, etc.).
*   **Call to Action:** Encourages users to try the cloud version or start with the quick start instructions.
*   **Comprehensive Examples Section:** Showcase the tool's versatility.
*   **MCP Section Expansion:** Provides more context and code examples for MCP integration, which could increase engagement.
*   **Roadmap and Vision:**  Provides context for current and future developments to keep users up-to-date.
*   **Clearer Instructions:** The installation and quick start instructions are emphasized and streamlined.
*   **Included links back to the source repo.**
*   **Swag and Citation sections** Included to entice contributors and properly recognize the project.
*   **Images are included with alt text** To increase accessibility.