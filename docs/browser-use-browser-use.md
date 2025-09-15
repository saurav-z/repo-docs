<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Browser Use: Automate Your Browser with AI 🤖

**Effortlessly control your browser using natural language with Browser Use, enabling AI-driven web automation.**  ([See the original repository](https://github.com/browser-use/browser-use))

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/browser-use/browser-use?lang=de) | 
[Español](https://www.readme-i18n.com/browser-use/browser-use?lang=es) | 
[français](https://www.readme-i18n.com/browser-use/browser-use?lang=fr) | 
[日本語](https://www.readme-i18n.com/browser-use/browser-use?lang=ja) | 
[한국어](https://www.readme-i18n.com/browser-use/browser-use?lang=ko) | 
[Português](https://www.readme-i18n.com/browser-use/browser-use?lang=pt) | 
[Русский](https://www.readme-i18n.com/browser-use/browser-use?lang=ru) | 
[中文](https://www.readme-i18n.com/browser-use/browser-use?lang=zh)

<br/>

Want to get started right away? Try our <b>[cloud](https://cloud.browser-use.com)</b> for faster, scalable, and stealth-enabled browser automation.

## Key Features:

*   **AI-Powered Automation:** Control your browser with simple natural language instructions.
*   **Web Scraping & Data Extraction:** Easily scrape data from websites.
*   **Form Filling & Automation:** Automate form submissions and other repetitive tasks.
*   **Integration with LLMs:** Seamlessly integrates with large language models like GPT-4.
*   **Model Context Protocol (MCP) Support:** Compatible with Claude Desktop and other MCP clients for extended capabilities.
*   **Cloud Deployment:**  Option to use a cloud service for easier setup, scalability, and stealth.
*   **Robust Testing:** Ensure the reliability of your agents with automated testing.

**🚀 Always use the latest version!** 
```bash
uv pip install --upgrade browser-use
```

## Quickstart Guide

**Prerequisites:** Python 3.11+ and a browser (Chrome or Chromium recommended).

Install Browser Use:

```bash
uv pip install browser-use
```

Install Chromium (if you don't have it):

```bash
uvx playwright install chromium --with-deps --no-shell
```

Example Usage:

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

Remember to add your API keys (e.g., `OPENAI_API_KEY=`) to your `.env` file.  Explore the [documentation 📕](https://docs.browser-use.com) for more configurations and options.

## Demos & Use Cases

Explore these examples to see Browser Use in action:

*   **Shopping:** [Add grocery items to cart and checkout](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py).
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
    
*   **LinkedIn to Salesforce:**  Add your latest LinkedIn follower to your leads in Salesforce.
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application:**  Find ML jobs, save them to a file, and start applying:  [find_and_apply_to_jobs.py](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py).
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

*   **Document Creation:** Write a letter in Google Docs to your Papa, thanking him for everything, and save the document as a PDF: [real_browser.py](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py).
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Hugging Face Model Search:** Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file: [save_to_file_hugging_face.py](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py).
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

For more inspiration, check out the [examples](examples) folder and the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.  Join the [Discord](https://link.browser-use.com/discord) to show off your projects!

## Model Context Protocol (MCP) Integration

Browser Use supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling seamless integration with clients like Claude Desktop.

### Use as MCP Server with Claude Desktop

Configure Claude Desktop:

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

Connect to and use tools from multiple MCP servers:

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

Refer to the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

## Vision

Our vision is simple:  **Tell your computer what to do, and watch it execute.**

## Roadmap

### Agent

*   [ ] Make agent 3x faster
*   [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   [ ] Enable interaction with all UI elements
*   [ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows

*   [ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience

*   [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization

*   [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome your contributions!  Please open issues for bugs or feature requests.  For documentation contributions, see the `/docs` folder.

## 🧪 Robust Agent Testing

Ensure the reliability of your Browser Use agents with our CI system:

*   **Add your task:** Create a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:** Your tasks will be automatically executed and evaluated on every update.

## Local Setup

Learn more about local development with the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Important:**  `main` is the primary development branch. For production, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser Use pride!  Check out our [Merch store](https://browsermerch.com). Great contributors will receive free swag 👀.

## Citation

If you use Browser Use in your research or projects, please cite this repository:

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