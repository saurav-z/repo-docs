<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

<div align="center">
  <a href="https://github.com/gregpr07/browser-use"><img src="https://img.shields.io/github/stars/gregpr07/browser-use?style=social" alt="GitHub stars"></a>
  <a href="https://link.browser-use.com/discord"><img src="https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://cloud.browser-use.com"><img src="https://img.shields.io/badge/Cloud-☁️-blue" alt="Cloud"></a>
  <a href="https://docs.browser-use.com"><img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Documentation"></a>
  <a href="https://x.com/intent/user?screen_name=gregpr07"><img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Twitter Follow (Gregor)"></a>
  <a href="https://x.com/intent/user?screen_name=mamagnus00"><img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Twitter Follow (Magnus)"></a>
  <a href="https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341" alt="Weave Badge"></a>
</div>

**Browser Use empowers you to control your browser with the power of AI, opening the door to limitless automation possibilities.**

Key Features:

*   **AI-Driven Browser Automation:** Automate tasks like web scraping, form filling, and more using natural language prompts.
*   **Easy Setup:** Get started quickly with a simple pip install and a few lines of code.
*   **Cloud Integration:** Utilize the hosted version for instant browser automation without setup.
*   **Model Context Protocol (MCP) Support:** Integrate with MCP-compatible clients like Claude Desktop for extended functionality.
*   **Interactive CLI:** Test and interact with Browser Use through a user-friendly command-line interface.
*   **Extensive Documentation & Examples:**  Find detailed guides and a wealth of examples to kickstart your projects.
*   **Robust Testing:** Ensure the reliability of your tasks by running them in our CI on every update.
*   **Community & Support:** Join the vibrant community on [Discord](https://link.browser-use.com/discord) and the [awesome-prompts](https://github.com/browser-use/awesome-prompts) repo for inspiration and collaboration.

## Getting Started

### Installation

Install the Browser Use package using pip:

```bash
pip install browser-use
```

Install the browser (e.g., Chromium):

```bash
playwright install chromium --with-deps --no-shell
```

### Example Usage

Here's a quick example to get you started:

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

Remember to add your API keys to a `.env` file. Refer to the [documentation 📕](https://docs.browser-use.com) for settings, models, and more.

## Cloud Version

Skip the setup and try the [hosted version](https://cloud.browser-use.com) of Browser Use for immediate browser automation.

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

Explore real-world examples of Browser Use in action:

*   **AI Did My Groceries:** Add grocery items to cart and checkout.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce:** Add latest LinkedIn follower to leads in Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Find & Apply to Jobs:** Read my CV & find ML jobs, save them to a file, and then start applying for them.

    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

*   **Write a Letter:** Write a letter in Google Docs and save as PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Hugging Face Models:** Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

## More Examples

Discover even more inspiring projects and examples in the [examples](examples) folder.  Share your creations and find inspiration in the [Discord](https://link.browser-use.com/discord).

## Vision

Automate your digital world with simple instructions.

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

We welcome contributions! Please open issues for bugs or feature requests. For documentation contributions, explore the `/docs` folder.

## 🧪 Robust Agent Tasks

Ensure the reliability of your AI agents by integrating your tasks into our CI process.

*   **Add Your Task:** Place a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic Validation:** Your tasks are automatically executed and assessed based on your defined criteria with every update.

## Local Setup

For comprehensive information on local setup and development, please refer to the [local setup 📕](https://docs.browser-use.com/development/local-setup) guide.

**Important:** The `main` branch represents the primary development branch with frequent changes. For production use, consider installing a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Want to show off your Browser Use swag? Check out our [Merch store](https://browsermerch.com).

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

<div align="center">
<img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/>

  <div align="center">
    <a href="https://x.com/intent/user?screen_name=gregpr07"><img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Twitter Follow (Gregor)"></a>
    <a href="https://x.com/intent/user?screen_name=mamagnus00"><img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Twitter Follow (Magnus)"></a>
  </div>
</div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
</div>

<a href="https://github.com/browser-use/browser-use">Back to the Top</a>