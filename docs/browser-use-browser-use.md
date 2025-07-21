<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

**Effortlessly control your web browser using the power of AI with Browser Use.**

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

🔗 **[Visit the Browser Use GitHub Repository](https://github.com/browser-use/browser-use)**

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language prompts, enabling complex web tasks.
*   **Easy Integration:** Simple installation and setup using pip.
*   **Cloud-Based Option:** Quickly get started with a hosted version.  **[Try the cloud ☁︎](https://cloud.browser-use.com)**.
*   **Model Context Protocol (MCP) Support:** Integrate with Claude Desktop and other MCP-compatible clients for extended capabilities.
*   **Interactive CLI:** Test and experiment with browser-use using the interactive CLI.
*   **Extensive Documentation:** Comprehensive documentation available for detailed usage and customization.
*   **Robust Testing:**  Integrate your tasks into our CI and benefit from automated validation.
*   **Active Community:** Join the Discord community to share projects and get support.

## Getting Started

### Installation

Install the library using pip:

```bash
pip install browser-use
```

Install browser dependencies (Playwright):

```bash
playwright install chromium --with-deps --no-shell
```

### Example Usage

Here's a simple example of how to use Browser Use:

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

Remember to add your API keys to a `.env` file.

### Interactive CLI

Use the interactive CLI for testing:

```bash
pip install "browser-use[cli]"
browser-use
```

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

*   **AI Did My Groceries:**  [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce Integration:**  ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   **Find and Apply to Jobs:**  ![Find and Apply to Jobs](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)
*   **Write a Letter in Google Docs:**  ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **Find Models on Hugging Face:**  ![Find Models on Hugging Face](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## Explore More Examples

Find more examples and inspiration in the [examples](examples) folder and our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo. Join the [Discord](https://link.browser-use.com/discord) to share your projects.

## Vision & Roadmap

Our vision is to enable you to tell your computer what to do and have it get it done!

### Roadmap

*   **Agent:** Improve memory and planning capabilities, and reduce token consumption.
*   **DOM Extraction:** Enhance UI element detection and representation.
*   **Workflows:** Implement workflow recording and rerunning.
*   **User Experience:** Create templates, improve documentation, and boost speed.
*   **Parallelization:** Introduce parallel task execution.

## Contributing

We welcome contributions!  Please submit issues for bugs or feature requests.  Contribute to the docs in the `/docs` folder.

## 🧪 Robust Agent Testing

Enhance your agents by running them in our CI:

*   Add your task as a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   Tasks will automatically be validated during updates.

## Local Setup

For more information, see the [local setup 📕](https://docs.browser-use.com/development/local-setup).

Use a stable [versioned release](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Show off your Browser Use swag! Check out our [Merch store](https://browsermerch.com).  Good contributors get free swag 👀.

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
```
Key improvements:

*   **SEO Optimization:** Includes relevant keywords throughout the document.
*   **Clear Headings:**  Uses proper heading hierarchy for readability and search engine parsing.
*   **Concise Summarization:**  Condenses the information while keeping key details.
*   **One-Sentence Hook:**  Provides a strong initial statement to grab attention.
*   **Bulleted Key Features:** Highlights the key benefits of the library.
*   **Emphasis on Benefits:**  Focuses on *what* the user gains, not just *what* the library does.
*   **Links & Calls to Action:**  Includes clear calls to action, direct links and better context for those links.
*   **Structured for Scanning:**  Uses lists and short paragraphs for easy scanning.
*   **Improved Language:**  More active and engaging language.
*   **Consolidated Information:**  Repeated content (like MCP instructions) is organized more efficiently.
*   **Stronger Roadmap:**  Clearer and more focused on the value proposition.
*   **Complete and Detailed:** Includes everything from the original and adds more organization.
*   **Markdown Formatting:** Properly formatted markdown for GitHub.
*   **Added "Vision" Section:** Added to address the "Vision" section that was missed.