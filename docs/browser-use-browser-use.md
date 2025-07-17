<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Browser Use: Empower AI to Control Your Browser 🤖

**Effortlessly automate your browser with AI using Browser Use, enabling you to interact with the web through natural language.**

[GitHub Repository](https://github.com/browser-use/browser-use)

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Powered Automation:** Control your browser using natural language prompts.
*   **Easy Integration:** Simple Python integration with a clean API.
*   **Cloud-Ready:** Try it instantly with our hosted cloud version.
*   **MCP Integration:** Seamlessly integrate with the Model Context Protocol for extended capabilities.
*   **Robust Testing:** Automated validation of your tasks within our CI.
*   **Extensive Documentation:** Comprehensive documentation to get you started and help you customize.
*   **Active Community:** Join our Discord for support, examples, and to share your projects.

## Quick Start

### Installation

Install using pip:

```bash
pip install browser-use
```

Install the browser dependencies:

```bash
playwright install chromium --with-deps --no-shell
```

### Basic Usage

Here's a simple example to get you started:

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

**Important:** Add your API keys for the providers you want to use to your `.env` file.
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

For more details, including more configuration options and detailed usage examples, see the [documentation 📕](https://docs.browser-use.com).

### Testing & CLI

*   **Web UI & Desktop App:** Test your tasks using the [Web UI](https://github.com/browser-use/web-ui) or [Desktop App](https://github.com/browser-use/desktop).
*   **Interactive CLI:**  Explore and test Browser Use with the interactive CLI.

```bash
pip install "browser-use[cli]"
browser-use
```

## MCP Integration

Browser-use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), providing seamless integration with MCP-compatible clients.

### Using Browser Use as an MCP Server with Claude Desktop

Configure Claude Desktop to access browser automation tools:

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

### Connecting External MCP Servers to Browser-Use

Extend agent capabilities by connecting to multiple external MCP servers:

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

### Examples of What You Can Do with Browser Use

*   **[Shopping](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py):** Add grocery items to cart and checkout.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce:** Add your latest LinkedIn follower to your leads in Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **[Job Application](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py):** Read your CV & find ML jobs, save them to a file, and then start applying for them in new tabs.

    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
*   **[Google Docs](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py):** Write a letter in Google Docs and save the document as a PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **[Hugging Face](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py):**  Look up models on Hugging Face, sort them and save the top 5.

    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

## Vision

**Make your computer execute commands based on your spoken instructions.**

## Roadmap

### Agent
*   [ ] Improve agent memory to handle +100 steps
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

We welcome contributions! Please feel free to open issues for bugs or feature requests. To contribute to the documentation, check out the `/docs` folder.

## 🧪 Robust Agent Testing

To make your agents more reliable, run your tasks in our CI—automatically, on every update!

-   **Add Your Task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
-   **Automatic Validation:** Every time we push updates, your task will be run by the agent and evaluated based on your criteria.

## Local Setup

To get started, see the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:** `main` is the primary development branch. For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Want to show off your Browser Use swag? Check out our [Merch store](https://browsermerch.com). Good contributors may receive free swag 👀.

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