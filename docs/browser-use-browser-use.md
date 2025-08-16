<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Browser Use empowers your AI agents to control and interact with web browsers, automating tasks with ease.**

[**Explore the Browser Use Repository**](https://github.com/browser-use/browser-use)

## Key Features

*   **AI-Powered Automation:** Enables AI agents to perform actions within web browsers.
*   **Easy Integration:** Simple Python installation and setup.
*   **Cloud-Based Option:** Try the hosted version for instant browser automation.
*   **Model Context Protocol (MCP) Support:** Integrates with Claude Desktop and other MCP-compatible clients.
*   **Interactive CLI:** Test functionality using a built-in command-line interface.
*   **Robust Testing:** Automated task validation through CI integration.
*   **Extensive Documentation:** Comprehensive documentation to guide you.
*   **Active Community:** Join the Discord for support, project sharing, and inspiration.

## Quick Start

### Installation

```bash
pip install browser-use
```

### Browser Setup

```bash
playwright install chromium --with-deps --no-shell
```

### Example Usage

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

*   Remember to add your API keys to your `.env` file.  See the [documentation 📕](https://docs.browser-use.com) for details on settings and models.

### Interactive CLI

```bash
pip install "browser-use[cli]"
browser-use
```

## Model Context Protocol (MCP) Integration

Browser Use seamlessly integrates with the Model Context Protocol (MCP) to provide enhanced functionality.

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
*   For more details, see the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Demos

### AI Did My Groceries

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

### LinkedIn to Salesforce

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

### Find and Apply to Jobs

![Find and Apply to Jobs](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)

### Letter to Papa

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

### Save to File Hugging Face

![Save to File Hugging Face](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## Explore More Examples

*   Browse the [examples](examples) folder.
*   Join the [Discord](https://link.browser-use.com/discord) to share your projects and get inspired.
*   Check out the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting ideas.

## Vision

Browser Use aims to make browser automation intuitive and powerful, allowing users to control the web with simple instructions.

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

We welcome contributions!  Please open issues for bugs or feature requests.  Contribute to the docs in the `/docs` folder.

## Robust Agent Testing

Automated task validation is available through our CI.

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Your task will be run and evaluated with every update.

## Local Setup

See the [local setup 📕](https://docs.browser-use.com/development/local-setup) in the documentation for more information.

*   Install a [versioned release](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Show off your Browser Use swag!  Check out our [Merch store](https://browsermerch.com).  Good contributors receive free swag 👀.

## Citation

If you use Browser Use in your research or projects, cite it using:

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
Key improvements and explanations:

*   **SEO-optimized hook:**  The one-sentence hook is placed right after the title, grabbing attention and telling the user *what* Browser Use does.
*   **Clear Headings:**  Uses clear and concise headings for better readability and organization.
*   **Bulleted Key Features:** Uses a bulleted list to highlight the main features.
*   **Concise Language:**  Avoids jargon and uses straightforward language.
*   **Call to Action:**  Uses calls to action like "Explore the Browser Use Repository" to encourage engagement.
*   **Focus on Benefits:**  The descriptions emphasize the *benefits* of using Browser Use (e.g., "AI-Powered Automation," "Easy Integration").
*   **Complete and Accurate:**  Keeps all the original content while enhancing the presentation.
*   **Markdown Formatting:** Uses proper Markdown for better rendering.
*   **Direct Links:**  Makes sure to link to the original repo.
*   **Removed Unnecessary Repetition:** Eliminated redundant phrases.
*   **Improved Structure for Clarity:** Enhanced the layout to make it easy to scan and understand.
*   **Added Context:** Clarified parts of the code samples for a more user-friendly experience.
*   **Refined roadmap and vision:** Kept the core elements while organizing it.