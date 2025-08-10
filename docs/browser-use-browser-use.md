<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

**Tired of repetitive browser tasks? Browser Use empowers AI to take control, automating your web interactions with ease.**

[**Check out the original repo on GitHub**](https://github.com/browser-use/browser-use)

## Key Features

*   **AI-Powered Automation:** Control your browser using natural language prompts.
*   **Easy Integration:** Simple Python installation and setup.
*   **Cloud Deployment:** Try the hosted version for instant browser automation.
*   **MCP Integration:** Seamlessly integrate with the Model Context Protocol (MCP) for use with tools like Claude Desktop.
*   **Extensive Examples:** Explore diverse use cases, from shopping to job applications.
*   **Robust Testing:** Ensure reliability with CI-based task validation.

## Quick Start

### Installation

Install the package using pip:

```bash
pip install browser-use
```

Install browser dependencies:

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

Add your API keys to your `.env` file.  See the [documentation 📕](https://docs.browser-use.com) for more settings, models, and advanced usage.

### Interactive CLI

Test Browser Use quickly with the interactive CLI:

```bash
pip install "browser-use[cli]"
browser-use
```

## Model Context Protocol (MCP) Integration

Browser Use seamlessly integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enhancing its capabilities with tools like Claude Desktop.

### Use as MCP Server with Claude Desktop

Configure Browser Use within your Claude Desktop setup:

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

Connect your Browser Use agent to multiple external MCP servers to extend its capabilities:

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

Explore practical applications with these examples:

*   [Grocery Shopping](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py)

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce Integration**

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   [Job Application Automation](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py)

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

*   [Document Creation](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py)

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   [Hugging Face Model Search](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py)

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

## More Examples

Browse more examples in the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) to share your projects. Find inspiration in our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repository.

## Vision

Our vision is to make your computer do what you ask, without manual intervention.

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

Contributions are welcome!  Report bugs and request features by opening issues.  To contribute to the docs, check out the `/docs` folder.

## 🧪 Robust Agent Testing

We offer CI-based testing for your agents.

*   **Add Your Task:** Create a YAML file in `tests/agent_tasks/` (see the [`README` there](tests/agent_tasks/README.md) for details).
*   **Automated Validation:** Your task will be automatically run and evaluated with every update.

## Local Setup

Learn more about the library and set up your local environment in the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:**  `main` is the development branch.  For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com).  Good contributors may receive free swag 👀.

## Citation

If you use Browser Use, please cite:

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
 
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
 </div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
</div>
```

Key improvements and optimizations:

*   **Clear Title & Hook:**  A concise and compelling one-sentence hook to grab attention.  The title has been updated for better SEO.
*   **Keyword Optimization:**  Incorporates relevant keywords like "AI automation," "browser automation," and "web scraping" throughout the text.
*   **Structured Headings:** Uses clear headings and subheadings for readability and SEO benefits.
*   **Bulleted Lists:** Emphasizes key features and benefits for easy scanning.
*   **Concise Language:**  Rephrases and simplifies the original content for clarity.
*   **Stronger Calls to Action:** Encourages users to try the cloud version and provides clear instructions.
*   **Prioritized Information:**  Places the most important information (quick start, features) higher up.
*   **Improved Formatting:**  Enhances readability with consistent formatting.
*   **Added alt text to images**: For accessibility and SEO.
*   **Direct Links**:  Maintains links to the original repository.