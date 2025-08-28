<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: AI-Powered Browser Automation</h1>

<div align="center">
  <p><b>Effortlessly automate your browser with AI, streamlining tasks and unlocking new possibilities.</b></p>
</div>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Driven Automation:** Control your browser using natural language prompts.
*   **Cloud-Ready:** Access a faster, scalable, and stealth-enabled browser automation solution with our [cloud](https://cloud.browser-use.com).
*   **Easy Setup:** Quick installation via pip.
*   **Model Context Protocol (MCP) Integration:** Integrates with Claude Desktop and other MCP-compatible clients.
*   **Extensive Examples:** Explore a variety of use cases and integrations.
*   **Robust Agent Tasks:** Easily validate your agent tasks.
*   **Customizable & Extensible:** Build with customizable workflows and connect to external services using MCP.

## Quick Start

Install using pip:

```bash
pip install browser-use
```

Install Chromium (if you don't have it) with playwright's install shortcut:

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

Remember to add your API keys to your `.env` file. Check out the [documentation 📕](https://docs.browser-use.com) for more details.

## Demos

<br/><br/>

[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

Prompt: Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## More Examples & Community

Explore more examples in the [examples](examples) folder and join the [Discord](https://link.browser-use.com/discord) to share your projects. Get inspiration from our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## Model Context Protocol (MCP) Integration

Browser-use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling integration with Claude Desktop and other MCP-compatible clients.

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

This gives Claude Desktop access to browser automation tools for web scraping, form filling, and more.

### Connect External MCP Servers to Browser-Use Agent

Browser-use agents can connect to multiple external MCP servers to extend their capabilities:

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

## Vision

Our goal is to enable users to simply tell their computer what to do, and have it executed.

## Roadmap

### Agent

*   \[ ] Make agent 3x faster
*   \[ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   \[ ] Enable interaction with all UI elements
*   \[ ] Improve state representation for UI elements so that any LLM can understand what's on the page

### Workflows

*   \[ ] Let user record a workflow - which we can rerun with browser-use as a fallback

### User Experience

*   \[ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.

### Parallelization

*   \[ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions!  Please open issues for bug reports and feature requests. To contribute to the documentation, check out the `/docs` folder.

## 🧪 Robust Agent Tasks

Ensure your tasks run smoothly with our automatic CI validation!

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:**  Our CI will run your task and evaluate it based on your criteria.

## Local Setup

To learn more about the library, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

`main` is the primary development branch. For production, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com). Great contributors can receive swag for free 👀.

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
  <br/>
  <a href="https://x.com/intent/user?screen_name=gregpr07"><img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Follow Gregor on Twitter"></a>
  <a href="https://x.com/intent/user?screen_name=mamagnus00"><img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Follow Magnus on Twitter"></a>
</div>

<div align="center">
  Made with ❤️ in Zurich and San Francisco
</div>
```

Key improvements and SEO considerations:

*   **Clear Headline & Hook:**  A compelling one-sentence summary to grab attention.  Also includes the primary keyword: "Browser Automation"
*   **Keyword-Rich Introduction:** Introduces the concept and core benefit of AI-powered automation.
*   **Structured Headings:**  Organized content with clear headings for readability and SEO.
*   **Bulleted Key Features:**  Highlights the benefits using bullet points.
*   **Strong Call to Action:** Encourages users to try the product.
*   **Internal Linking:**  Links to documentation, examples, and the cloud offering to improve navigation and SEO.
*   **External Links:** Links to the Discord, Cloud, and Twitter accounts.
*   **Concise Language:**  Streamlines the information for better engagement.
*   **Relevant Images:** Visuals to break up text and improve appeal.
*   **Roadmap Section:** Showcases future development which can attract users.
*   **Contribution Guidelines:** Explains how to contribute, which can bring in more contributors.
*   **Clear Citation:**  Provides guidance for citation purposes.
*   **Social Media Links:**  Include Twitter profiles.
*   **Overall:** Improves readability and provides a much better user experience, improving discoverability for search engines.
*   **SEO Focus:** Uses relevant keywords like "AI," "browser automation," and "automation" naturally throughout the content.
*   **Original Repo Link:**  Includes a backlink to the original repo.