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

**Browser Use allows you to control your web browser with AI, automating tasks and streamlining your workflow.**  This project, and its source code, can be found on [GitHub](https://github.com/browser-use/browser-use).

🌤️ Need a quick start? Try our <b>[cloud](https://cloud.browser-use.com)</b> for fast, scalable, and stealth-enabled browser automation!

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language instructions.
*   **Versatile Use Cases:** Automate tasks like online shopping, data entry, job applications, and document creation.
*   **Easy Integration:**  Simple Python integration with pip, and API key configuration.
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients.
*   **Robust Examples:**  Comprehensive examples demonstrating various automation capabilities.
*   **Cloud Solution:** Fast, scalable, and stealth-enabled browser automation with the [cloud](https://cloud.browser-use.com).
*   **Active Community:** Join the [Discord](https://link.browser-use.com/discord) to connect and show off your project.

## Quickstart

### Installation

Install Browser Use using pip:

```bash
pip install browser-use
```

If you don't already have Chrome or Chromium installed, you can also download the latest Chromium using playwright's install shortcut:

```bash
uvx playwright install chromium --with-deps --no-shell
```

### Run Your First Agent

Here's a basic example to get you started:

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

*Remember to add your API keys to a `.env` file:*

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

For more details on settings, models, and configurations, see the [documentation 📕](https://docs.browser-use.com).

## Demos

Explore these examples to see Browser Use in action:

*   [Shopping](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   LinkedIn to Salesforce:  Add your latest LinkedIn follower to your leads in Salesforce.
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   [Job Application](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py):  Find and apply for ML jobs from your CV.
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
*   [Google Docs](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to your Papa and save as a PDF.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   [Hugging Face](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py):  Search for models on Hugging Face, save the top 5 to a file.
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

For more examples, see the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) to share your projects.  Also, check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting inspiration.

## Model Context Protocol (MCP) Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), expanding its functionality with MCP-compatible clients.

### Use as MCP Server with Claude Desktop

Integrate Browser Use into your Claude Desktop configuration:

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

This allows Claude Desktop to leverage Browser Use's automation tools.

### Connect External MCP Servers to Browser-Use Agent

Extend Browser Use's capabilities by connecting to external MCP servers:

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

See the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for further details.

## Vision

Command your computer and watch it perform tasks with Browser Use!

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

We welcome your contributions!  Please submit issues for bugs or feature requests.  Contribute to the docs by editing the `/docs` folder.

## 🧪 How to make your agents robust?

Ensure your tasks are validated automatically with our CI.

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Every time we push updates, your task will be run by the agent and evaluated using your criteria.

## Local Setup

For detailed information on setting up your local environment, consult the [local setup 📕](https://docs.browser-use.com/development/local-setup).

*Important: The `main` branch is the primary development branch.*  For stable releases, install a [versioned release](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Show off your Browser Use support!  Check out our [Merch store](https://browsermerch.com).  Good contributors may receive free swag!

## Citation

If you use Browser Use in your research or projects, please cite:

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