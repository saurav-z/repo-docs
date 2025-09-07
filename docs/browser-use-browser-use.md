<!-- Improved & Summarized README -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

**Unlock the power of AI to control your web browser, automating tasks and workflows with natural language.**  Explore the [Browser Use GitHub Repository](https://github.com/browser-use/browser-use) to get started.

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

## Key Features

*   **AI-Powered Automation:** Control your browser with natural language prompts, enabling complex workflows.
*   **Cloud Deployment:** Use our cloud for faster, scalable browser automation.
*   **Model Context Protocol (MCP) Integration:**  Integrate with Claude Desktop and other MCP-compatible clients.
*   **Extensive Examples:** Explore a diverse range of use cases.
*   **Easy Setup:**  Simple installation with Python's pip.
*   **Robustness:**  Automated CI testing to validate your tasks.

🌤️ Want to skip the setup? Use our <b>[cloud](https://cloud.browser-use.com)</b> for faster, scalable, stealth-enabled browser automation!

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

1.  **Install:**
    ```bash
    pip install browser-use
    ```

2.  **Install Chromium (if you don't have it already):**
    ```bash
    uvx playwright install chromium --with-deps --no-shell
    ```

3.  **Run a simple agent:**
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

4.  **Configure API Keys:**  Add your API keys (e.g., `OPENAI_API_KEY=`) to a `.env` file.

5.  **Explore the Documentation:**  For more settings and model options, see the [documentation 📕](https://docs.browser-use.com).

## Demos & Use Cases

<br/><br/>

**Grocery Shopping:** Add grocery items to cart and check out.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

**LinkedIn to Salesforce:**  Add your latest LinkedIn follower to your leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

**Job Application Automation:** Read your CV, find ML jobs, save them, and start applying.

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

**Document Generation:** Write a letter in Google Docs and save it as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

**Hugging Face Model Search:** Look up models with a specific license and save the top results to a file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## More Examples & Community

*   **Explore:** Browse the [examples](examples) folder for more inspiration.
*   **Connect:** Join the [Discord](https://link.browser-use.com/discord) to share your projects and get help.
*   **Prompting:** Check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## Model Context Protocol (MCP) Integration

Browser-use seamlessly integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enhancing its capabilities with external services.

### Use as MCP Server with Claude Desktop

Configure Claude Desktop to use browser-use as an MCP server:

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

Expand your agent's functionality by connecting to external MCP servers:

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

Enable intuitive, AI-driven browser automation.

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

We welcome contributions!  Please open issues or submit pull requests.  For documentation contributions, see the `/docs` folder.

## 🧪 Automated Testing

Run your tasks automatically with CI:

*   **Add a task:**  Create a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Validation:** Your task will be automatically run and evaluated with every update.

## Local Setup

To learn more about the library, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).

Install a stable [versioned release](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Show off your Browser-use pride! Check out our [Merch store](https://browsermerch.com). Great contributors get free swag 👀.

## Citation

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