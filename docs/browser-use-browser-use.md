<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Control Your Browser with AI 🤖</h1>

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

## 🚀 Introduction

**Browser Use empowers you to automate your browser with AI, letting you execute complex tasks with natural language commands.**  This open-source project (view the original repo [here](https://github.com/browser-use/browser-use)) allows you to control your web browser programmatically using the power of AI.

## ✨ Key Features

*   **AI-Powered Automation:** Control your browser using natural language prompts.
*   **Easy Setup:** Simple installation with pip and straightforward code examples.
*   **Versatile Use Cases:** Automate tasks like web scraping, form filling, and more.
*   **Cloud Integration:**  Use the [cloud](https://cloud.browser-use.com) for scalable and stealth automation.
*   **MCP Integration:** Integrates with Model Context Protocol (MCP) for expanded capabilities.
*   **Robustness:** Tasks can be automatically validated in CI.

## 📣 Announcements

### 🎉 OSS Twitter Hackathon: #nicehack69

To celebrate hitting **69,000 GitHub ⭐**, we're launching a Twitter-first hackathon with a **$6,900 prize pool**! Show us the future of browser-use agents!

**Deadline: September 10, 2025**

**[🚀 Join the hackathon →](https://github.com/browser-use/nicehack69)**

<div align="center">
<a href="https://github.com/browser-use/nicehack69">
<img src="./static/NiceHack69.png" alt="NiceHack69 Hackathon" width="600"/>
</a>
</div>

## 📦 Installation and Quickstart

**Update to the latest version:**

```bash
pip install --upgrade browser-use
```

**Install Dependencies (if needed):**

```bash
uvx playwright install chromium --with-deps --no-shell
```

**Run a simple agent:**

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

*Remember to add your API keys to your `.env` file.*
*   Refer to the [documentation 📕](https://docs.browser-use.com) for settings, models, and advanced usage.

## 💡 Demos

Explore the potential of Browser Use with these examples:

<br/><br/>

*   [Grocery Shopping](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py):  Automate grocery shopping (see video below)

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

*   **LinkedIn to Salesforce:**  Add your latest LinkedIn follower to Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

*   [Job Application](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Find and apply for ML jobs based on your CV.

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

*   [Letter to Papa](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to your Papa.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

*   [Hugging Face Model Search](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Find and save models with specific licenses on Hugging Face.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## 📚 More Examples

For more examples, check out the [examples](examples) folder and the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo or join the [Discord](https://link.browser-use.com/discord) community to showcase your projects.

## ⚙️ MCP Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for expanded capabilities with compatible clients.

### Use as MCP Server with Claude Desktop

Integrate Browser Use with Claude Desktop for enhanced functionality.

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

Extend agent capabilities by connecting to multiple external MCP servers.

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

See the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

## 💡 Vision

**Our goal: To enable you to simply tell your computer what you want done and have it executed.**

## 🗺️ Roadmap

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

## 🤝 Contributing

We welcome contributions! Report bugs and request features by opening issues. For documentation contributions, check out the `/docs` folder.

## ✅ Robustness Testing

Improve task reliability using our CI.

*   **Add Your Task:** Add a YAML file in `tests/agent_tasks/` (see [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic Validation:** Your task is run and evaluated automatically with every update.

## 💻 Local Setup

Explore the library further by setting up a [local environment 📕](https://docs.browser-use.com/development/local-setup).

**Note:** `main` is the primary development branch. For production use, use a stable [versioned release](https://github.com/browser-use/browser-use/releases).

---

## 🛍️ Swag

Show your support with Browser Use swag! Visit our [Merch store](https://browsermerch.com). Good contributors may receive free swag 👀.

## 📚 Citation

If you utilize Browser Use in your research or project, please cite it as follows:

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