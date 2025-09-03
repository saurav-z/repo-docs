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

**Supercharge your web interactions: Browser Use allows you to control your browser using the power of AI, automating complex tasks with ease.**

🌤️ Want to skip the setup? Use our [cloud](https://cloud.browser-use.com) for faster, scalable, stealth-enabled browser automation!

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language prompts.
*   **Cloud Integration:** Utilize a cloud service for easy deployment and scalability.
*   **Model Context Protocol (MCP) Support:** Integrate with tools like Claude Desktop for extended capabilities.
*   **Flexible Examples:** Pre-built examples showcase grocery ordering, Salesforce integration, job applications, and more.
*   **Open Source & Community Driven:** Benefit from a vibrant community and contribute to the project.

## 🎉 OSS Twitter Hackathon

We just hit **69,000 GitHub ⭐**!
To celebrate, we're launching **#nicehack69** — a Twitter-first hackathon with a **$6,900 prize pool**. Dream big and show us the future of browser-use agents that go beyond demos!

**Deadline: September 6, 2025**

**[🚀 Join the hackathon →](https://github.com/browser-use/nicehack69)**

<div align="center">
<a href="https://github.com/browser-use/nicehack69">
<img src="./static/NiceHack69.png" alt="NiceHack69 Hackathon" width="600"/>
</a>
</div>

## Quickstart

Easily install and start using Browser Use to automate your browser tasks.

**1. Installation:**

```bash
pip install browser-use
```

If you don't already have Chrome or Chromium installed, you can also download the latest Chromium using playwright's install shortcut:

```bash
uvx playwright install chromium --with-deps --no-shell
```

**2. Implement your Agent:**

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

**3. Set API Keys:**

Add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
```

For other settings, models, and more, check out the [documentation 📕](https://docs.browser-use.com).

## Demos

Explore the power of Browser Use through these examples:

*   **Grocery Ordering:** Add items to a cart and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce:**  Add your latest LinkedIn follower to your leads in Salesforce.
    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   **Job Application:** Read your CV & find ML jobs, save them to a file, and then start applying for them in new tabs.
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
*   **Create a Letter:** Write a letter in Google Docs and save the document as a PDF.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **Find and Save Hugging Face Models:** Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.
    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

## More Examples

Find more examples in the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) to share your own projects. You can also see our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting inspiration.

## Model Context Protocol (MCP) Integration

Browser Use seamlessly integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enhancing its capabilities with Claude Desktop and other MCP-compatible clients.

### Use as MCP Server with Claude Desktop

Configure Browser Use as an MCP server within Claude Desktop:

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

Extend Browser Use agent capabilities by connecting to multiple external MCP servers:

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

## Vision

*   Effortlessly automate web interactions through natural language.

## Roadmap

### Agent

*   Make agent 3x faster
*   Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   Enable interaction with all UI elements
*   Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows

*   Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience

*   Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization

*   Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions! Please feel free to open issues for bug reports or feature requests. To contribute to the docs, check out the `/docs` folder.

## 🧪 How to make your agents robust?

We offer to run your tasks in our CI—automatically, on every update!

-   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
-   **Automatic validation:** Every time we push updates, your task will be run by the agent and evaluated using your criteria.

## Local Setup

Learn more about the library with our [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Important:** `main` is the active development branch. For stable production use, please install a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser Use pride! Visit our [Merch store](https://browsermerch.com). Good contributors can get swag for free 👀.

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
Key improvements and SEO optimizations:

*   **Clear, Concise Title:**  "Browser Use: Automate Your Browser with AI" is more keyword-rich and descriptive.  Added the direct benefit ("Automate Your Browser").
*   **SEO-Friendly Headings:** Uses proper H1, H2, etc. headings for structure and SEO.
*   **Keyword Integration:**  Naturally incorporates keywords like "browser automation," "AI," and "automation" throughout.
*   **Benefit-Driven Hook:** Starts with a one-sentence hook highlighting the core value proposition.
*   **Bulleted Key Features:** Makes the features easily scannable and highlights key selling points.
*   **Concise Descriptions:** Improves the readability of each section.
*   **Call to Actions:** Includes clear call to actions like "Join the hackathon" and "Explore the examples."
*   **Comprehensive Coverage:** Summarizes all essential parts of the original README.
*   **Internal Linking:** Includes links to the documentation and the source code for enhanced navigation.
*   **Clean Formatting:** Uses Markdown for readability and structure.
*   **Focus on Value:** The language emphasizes the benefits users will receive from using the project.
*   **Clear roadmap**: Better structure, with some items edited to be more general for better understanding
*   **Concise:** Keeps the README focused and easy to scan.
*   **Added Links:** Added a link back to the original repo: [https://github.com/browser-use/browser-use](https://github.com/browser-use/browser-use)

This revised README is more compelling, SEO-optimized, and user-friendly.