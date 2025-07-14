<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

<p align="center"><b>Give your AI the power to interact with the web and get things done!</b></p>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**[View the original repository on GitHub](https://github.com/browser-use/browser-use)**

Browser Use empowers your AI agents to interact with and control web browsers, enabling automation for tasks like web scraping, form filling, and much more.

Key Features:

*   **AI-Powered Automation:** Automate browser tasks using the power of AI.
*   **Easy Integration:**  Simple setup and integration with Python.
*   **Cloud-Ready:** Skip setup with the [hosted cloud version](https://cloud.browser-use.com).
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients.
*   **Interactive CLI:** Test and experiment with the included interactive CLI.
*   **Robust Testing:** Contribute to the project by submitting your automation tasks for validation.

## Quick Start

Install using pip (requires Python>=3.11):

```bash
pip install browser-use
```

Install the browser dependencies (Chromium):

```bash
playwright install chromium --with-deps --no-shell
```

Create and run your agent:

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

Populate your `.env` file with API keys for your preferred AI providers.

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

For detailed information on additional settings, available models, and more, explore the comprehensive [documentation 📕](https://docs.browser-use.com).

### Testing and UI

*   **Web UI:** Test browser-use with the [Web UI](https://github.com/browser-use/web-ui).
*   **Desktop App:** Test browser-use using the [Desktop App](https://github.com/browser-use/desktop).
*   **Interactive CLI:** Use the `browser-use` interactive CLI (similar to `claude` code):

```bash
pip install "browser-use[cli]"
browser-use
```

## Model Context Protocol (MCP) Integration

Browser-use supports the Model Context Protocol (MCP), enabling integration with tools such as Claude Desktop.

### Browser-use as MCP Server

Configure Claude Desktop to use browser-use as an MCP server:

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

This provides Claude Desktop access to browser automation capabilities.

### Connecting External MCP Servers to Browser-Use Agent

Extend your agents with external MCP servers:

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

For more information, see the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Demos

Explore various use cases:

[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

Prompt: Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## More Examples

Find more examples and project inspiration in the [examples](examples) folder and the [Discord](https://link.browser-use.com/discord), or show off your own project! Check out the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for more prompting inspiration.

## Vision

Empowering users with the ability to command their computer to execute web-based tasks effortlessly.

## Roadmap

### Agent

*   \[ ] Improve agent memory to handle +100 steps
*   \[ ] Enhance planning capabilities (load website specific context)
*   \[ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   \[ ] Enable detection for all possible UI elements
*   \[ ] Improve state representation for UI elements so that all LLMs can understand what's on the page

### Workflows

*   \[ ] Let user record a workflow - which we can rerun with browser-use as a fallback
*   \[ ] Make rerunning of workflows work, even if pages change

### User Experience

*   \[ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
*   \[ ] Improve docs
*   \[ ] Make it faster

### Parallelization

*   \[ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

Contributions are welcome!  Please feel free to submit bug reports or feature requests.  For documentation contributions, see the `/docs` folder.

## 🧪 Agent Robustness Testing

To make your AI agents robust, submit your task in our CI tests!

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Your task will be run and evaluated on every update.

## Local Setup

Learn more by checking out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:** The `main` branch is the primary development branch. For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases) instead.

---

## Swag

Show off your Browser-use swag. Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag for free 👀.

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