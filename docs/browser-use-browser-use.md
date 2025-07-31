<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Browser Use: Automate Your Browser with AI 🤖

**Effortlessly control your browser with AI and unlock powerful automation capabilities.**

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Browser Automation:** Control your browser using natural language prompts.
*   **Easy Integration:** Seamlessly connect your AI agents with the browser.
*   **MCP Integration:** Supports the Model Context Protocol (MCP) for integration with tools like Claude Desktop.
*   **Hosted Cloud Version:**  Try the [hosted version ☁︎](https://cloud.browser-use.com) for instant browser automation without setup.
*   **Extensive Documentation:** Comprehensive [documentation 📕](https://docs.browser-use.com) to get you started.
*   **Interactive CLI:** Use the browser-use interactive CLI for quick testing.
*   **Robust Testing:** Run your tasks in our CI for automatic validation.
*   **Open Source:**  Explore, contribute, and build upon the [Browser-Use](https://github.com/browser-use/browser-use) project.

## Quick Start

1.  **Installation (with Python>=3.11):**

    ```bash
    pip install browser-use
    ```

2.  **Install Browser (Playwright):**

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

4.  **API Keys:** Add your API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) to a `.env` file.

## MCP Integration - Extending Agent Capabilities

Browser-use seamlessly integrates with the Model Context Protocol (MCP), enabling advanced workflows.

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

Extend your agent's capabilities by connecting to external MCP servers:

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

Explore real-world use cases with the browser-use demos:

*   **[AI Did My Groceries](https://www.youtube.com/watch?v=L2Ya9PYNns8):**  Adding groceries to a cart and checking out.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce:** Adding a LinkedIn follower to Salesforce leads.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Find and Apply to Jobs:**  Reading a CV, finding ML jobs, and applying.

    ![Find and Apply to Jobs](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)

*   **Write a Letter:** Writing a letter in Google Docs and saving it as a PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Find Models on Hugging Face:** Looking up models on Hugging Face and saving them to a file.

    ![Find Models on Hugging Face](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## More Examples & Inspiration

*   Explore the [examples](examples) folder for detailed use cases.
*   Join the [Discord](https://link.browser-use.com/discord) to share your projects and get inspired.
*   Check out the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting ideas.

## Vision

Empowering users by enabling AI to understand and execute instructions within the browser.

## Roadmap

### Agent

*   [ ] Improve agent memory for +100 steps
*   [ ] Enhance planning capabilities (load website specific context)
*   [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   [ ] Enable detection for all possible UI elements
*   [ ] Improve state representation for UI elements so that all LLMs can understand what's on the page

### Workflows

*   [ ] Let user record a workflow - which we can rerun with browser-use as a fallback
*   [ ] Make rerunning of workflows work, even if pages change

### User Experience

*   [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
*   [ ] Improve docs
*   [ ] Make it faster

### Parallelization

*   [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions!  Feel free to open issues or pull requests.

*   **Docs:** Contribute to the docs in the `/docs` folder.
*   **Testing:** Add your task with a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).

## Local Setup

Learn more about setting up your local environment: [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Important:** Use a [versioned release](https://github.com/browser-use/browser-use/releases) for production.

---

## Swag & Community

*   Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com).
*   Good contributors will receive swag for free 👀.

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