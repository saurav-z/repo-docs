<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

**Tired of manual browser tasks? Browser Use empowers you to control your browser with natural language.**  [Explore the original repo](https://github.com/browser-use/browser-use).

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

<br>

## Key Features

*   **AI-Powered Automation:** Control your browser using natural language prompts.
*   **Cloud Deployment Option:** Get started quickly with our [cloud](https://cloud.browser-use.com) service for scalable and stealth-enabled browser automation.
*   **Integration with LLMs:** Seamlessly integrates with leading Large Language Models (LLMs).
*   **Model Context Protocol (MCP) Support:** Supports integration with Claude Desktop and other MCP-compatible clients, extending functionality with external tools.
*   **Extensible with External MCP Servers:** Connect to multiple external MCP servers to enhance agent capabilities.
*   **Robustness Through Automated Testing:** Built-in CI to run your agent tasks automatically.

<br>

## Quick Start

Install Browser Use using pip:

```bash
pip install browser-use
```

If you don't have Chrome/Chromium, install it using Playwright:

```bash
uvx playwright install chromium --with-deps --no-shell
```

Run a simple agent:

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

Configure your API keys in a `.env` file:

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

For further customization and details, consult the comprehensive [documentation 📕](https://docs.browser-use.com).

## Demos

Here are some example use cases:

*   **[Shopping](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py):** Add groceries to cart and checkout.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
    
*   **LinkedIn to Salesforce Integration:** Automatically add your latest LinkedIn follower to your leads in Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
   
*   **[Job Application](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py):** Find and apply for ML jobs based on your CV.
    
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
    
*   **[Document Creation](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py):** Write and save a letter in Google Docs as a PDF.
   
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
    
*   **[Hugging Face Data Extraction](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py):** Look up models on Hugging Face and save data to a file.

    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br>

## More Examples & Resources

Explore more examples in the [examples](examples) folder. For inspiration and community support, join the [Discord](https://link.browser-use.com/discord). Check out the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting ideas.

<br>

## MCP Integration

Browser Use offers support for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling integration with tools like Claude Desktop:

*   **Use as MCP Server:** Integrate with Claude Desktop by adding the following to your Claude Desktop configuration:

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

*   **Connect External MCP Servers to Browser-Use Agent:** Extend your agent's capabilities:

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

For detailed MCP information, check out the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Vision

Empowering you to control your computer through natural language.

<br>

## Roadmap

### Agent
- [ ] Make agent 3x faster
- [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction
- [ ] Enable interaction with all UI elements
- [ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows
- [ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience
- [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization
- [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

<br>

## Contributing

We welcome contributions!  Report bugs or request features by opening issues. For documentation contributions, see the `/docs` folder.

<br>

## Robust Agent Tasks with CI

Ensure the reliability of your agent tasks with our built-in CI:

*   **Add Your Task:**  Create a YAML file in `tests/agent_tasks/`.  See the [`README there`](tests/agent_tasks/README.md) for setup details.
*   **Automated Validation:**  Your task will be automatically executed and assessed on every update.

<br>

## Local Setup

Learn more about setting up the project locally in the [local setup 📕](https://docs.browser-use.com/development/local-setup).

<br>

## Releases

`main` is the active development branch.  For production use, use a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

<br>

## Browser Use Swag!

Show off your support with our [Merch store](https://browsermerch.com)!  Good contributors may receive free swag! 👀

<br>

## Citation

If you use Browser Use in your research or projects, please cite it as follows:

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