<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Browser-Use: Unleash AI's Power in Your Browser 🤖

**Effortlessly automate your browser tasks with the power of AI.** [Visit the original repository](https://github.com/browser-use/browser-use).

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Automation:** Control your browser using natural language instructions.
*   **Easy Integration:** Simple installation with pip and straightforward setup.
*   **Cloud-Ready:** Try the hosted version for instant browser automation.  [Try the cloud ☁︎](https://cloud.browser-use.com).
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients.
*   **Extendable with MCP Servers:** Connect to external MCP servers to expand agent capabilities.
*   **Interactive CLI:**  Test and experiment with an interactive command-line interface.
*   **Robust Testing:**  Run your tasks in our CI—automatically, on every update!
*   **Extensive Documentation:** Comprehensive documentation to get you started.
*   **Active Community:** Engage with other users in our [Discord](https://link.browser-use.com/discord).

## Quick Start

### Installation

Install the library:

```bash
pip install browser-use
```

Install the browser:

```bash
playwright install chromium --with-deps --no-shell
```

### Basic Usage

Here's how to get started with a simple example:

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

### API Keys

Add your API keys to your `.env` file:

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

## Test with UI

You can test browser-use using its [Web UI](https://github.com/browser-use/web-ui) or [Desktop App](https://github.com/browser-use/desktop).

## Test with an interactive CLI

You can also use our `browser-use` interactive CLI (similar to `claude` code):

```bash
pip install "browser-use[cli]"
browser-use
```

## Model Context Protocol (MCP) Integration

Browser-use seamlessly integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

### Use as MCP Server with Claude Desktop

Configure Browser-use within Claude Desktop:

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

For further details, see the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Demos

### AI Did My Groceries

Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

### LinkedIn to Salesforce

Prompt: Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

### Find and Apply to Jobs

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

### Letter to Papa

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

### Hugging Face Models

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## More Examples

Explore more use cases in the [examples](examples) folder, and join the [Discord](https://link.browser-use.com/discord) to share your projects and get inspired.  Also check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## Vision

Empowering you to control your computer with natural language, making complex tasks simple.

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

We welcome contributions!  Please submit issues for bugs and feature requests. For documentation contributions, see the `/docs` folder.

## 🧪 How to make your agents robust?

Ensure your tasks are validated automatically by adding them to our CI system:

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Your task will be run and evaluated on every update.

## Local Setup

See the [local setup 📕](https://docs.browser-use.com/development/local-setup) for details.

**Note:** `main` is the primary development branch. For production, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com). Excellent contributors may receive free swag.

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