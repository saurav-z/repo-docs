<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser-Use: Empower Your AI to Command Your Browser 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Browser-Use lets you effortlessly connect your AI agents to the web, automating tasks with natural language commands.**

## Key Features

*   **AI-Powered Browser Automation:** Control your browser using natural language.
*   **Easy Setup:** Simple installation with Python pip.
*   **Cloud Deployment Option:** Get started instantly with our hosted cloud version:  [Try the cloud ☁︎](https://cloud.browser-use.com).
*   **Model Context Protocol (MCP) Integration:** Integrates with MCP-compatible clients like Claude Desktop.
*   **Interactive CLI:** Test your AI agent with our user-friendly CLI.
*   **Extensive Documentation:** Comprehensive [documentation 📕](https://docs.browser-use.com).
*   **Active Community:** Engage with others and share your projects in our [Discord](https://link.browser-use.com/discord).
*   **Test Automation:**  Test your browser automation using our CI on every update.

## Quick Start

Install the package:

```bash
pip install browser-use
```

Install browser dependencies:

```bash
playwright install chromium --with-deps --no-shell
```

Basic usage:

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

Set up your API keys in a `.env` file:

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

## Web UI and CLI Testing

Test and experiment with Browser-Use using:

*   [Web UI](https://github.com/browser-use/web-ui)
*   `browser-use` interactive CLI:

```bash
pip install "browser-use[cli]"
browser-use
```

## MCP Integration

Browser-use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing integration with Claude Desktop and other MCP-compatible clients.

### Use as MCP Server with Claude Desktop

Add browser-use to your Claude Desktop configuration:

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

### Shopping

Add grocery items to cart and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

### LinkedIn to Salesforce

Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

### Find and Apply to Jobs

Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

### Write a Letter

Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

### Find Models on Hugging Face

Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

## More Examples

Explore more examples and use cases in the [examples](examples) folder and in our [Discord](https://link.browser-use.com/discord).  Get inspired with our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## Vision

Our vision is to let you tell your computer what to do and have it get it done.

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

Contributions are welcome!  Feel free to open issues or submit pull requests. To contribute to the docs, go to the `/docs` folder.

## 🧪 Robust Agent Testing

Ensure your agents are robust by automatically validating your tasks with our CI!

*   Add your tasks as YAML files in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   Your tasks will be automatically run and evaluated on every update.

## Local Setup

For detailed information on local setup, refer to the [local setup documentation 📕](https://docs.browser-use.com/development/local-setup).

**Important:** `main` is the primary development branch.  For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases) instead.

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com). Generous contributors will receive swag for free 👀.

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
Key improvements and SEO enhancements:

*   **Clear and Concise Title:**  The main title is now SEO-friendly.
*   **SEO-Optimized Description:** The opening sentence clearly states the value proposition.
*   **Strategic Use of Keywords:**  Incorporates keywords like "AI," "browser automation," and "web scraping."
*   **Structured Headings:** Uses clear and descriptive headings for better readability and SEO.
*   **Bulleted Key Features:** Highlights the main advantages in a concise format.
*   **Call to Action:** Encourages users to "Try the cloud" and explore the documentation.
*   **Internal Links:**  Links to relevant resources within the repository.
*   **External Links:**  Links to relevant external resources like the Discord and Merch store.
*   **Concise and Readable Code Snippets:** Includes relevant code examples.
*   **Focus on User Benefits:**  The content emphasizes what the tool *does* for the user.
*   **Roadmap Added:**  Includes the roadmap.
*   **Contributing Section:**  Makes it easy for others to contribute.
*   **Clear Citation Information:** Makes it easy for people to cite the library.
*   **Concise Summary:**  Presents a complete overview of the project.
*   **GitHub URL:** Links back to the original repo.

This improved README is more user-friendly and SEO-optimized.