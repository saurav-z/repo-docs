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
[![Twitter Follow](https://img.shields/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Browser Use empowers you to control your web browser using the power of AI, automating tasks and streamlining your workflow.**  [Check out the original repo](https://github.com/browser-use/browser-use).

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language commands.
*   **Cloud Integration:**  Use the [cloud](https://cloud.browser-use.com) for faster, scalable, and stealth-enabled automation.
*   **Easy Setup:** Simple installation with `pip install browser-use`.
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients.
*   **Extensive Examples:** Explore use cases like shopping automation, LinkedIn integration, and job application assistance.
*   **Robustness Testing:** Automated task validation through CI for reliable agent performance.
*   **Open Source:** Actively maintained with community contributions welcome.
*   **Swag:**  Good contributors will receive swag for free 👀.

## Quick Start

1.  **Installation:**

    ```bash
    pip install browser-use
    ```

2.  **Install Chromium (if needed):**

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

4.  **API Key:**  Add your API keys for the provider you want to use to your `.env` file.

    ```bash
    OPENAI_API_KEY=
    ```

For more details, see the [documentation 📕](https://docs.browser-use.com).

## Demos

Explore the capabilities of Browser Use with these examples:

*   **Shopping Automation:**  Add grocery items to your cart and checkout. ([View Demo](https://www.youtube.com/watch?v=L2Ya9PYNns8))
*   **LinkedIn to Salesforce Integration:** Automate lead capture.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application:** Find and apply for jobs based on your CV.

    ![Job Application Demo](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)
*   **Document Generation:**  Write a letter in Google Docs and save it as a PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **Hugging Face Data Extraction:**  Find and save models from Hugging Face.

    ![Hugging Face Data Extraction](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## MCP Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

### Use as MCP Server with Claude Desktop

Integrate Browser Use with Claude Desktop:

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

Extend your agent's capabilities:

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

## Vision

Make your computer do what you say it to do.

## Roadmap

### Agent

*   \[ ] Make agent 3x faster
*   \[ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   \[ ] Enable interaction with all UI elements
*   \[ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows

*   \[ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience

*   \[ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization

*   \[ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

Contributions are welcome!  Please feel free to open issues or pull requests.  For documentation contributions, see the `/docs` folder.

## 🧪 Robustness Testing

Improve your agent's performance.

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Tasks run on every update and evaluated using your criteria.

## Local Setup

Learn more about setting up the library by checking out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

*`main`* is the primary development branch; for production, install a [versioned release](https://github.com/browser-use/browser-use/releases).

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