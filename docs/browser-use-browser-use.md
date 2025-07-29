<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: AI-Powered Browser Automation</h1>

**Effortlessly control your browser with the power of AI using Browser Use, enabling automation for a wide range of tasks.** ([See the original repo](https://github.com/browser-use/browser-use))

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

## Key Features

*   **AI-Driven Automation:**  Control your browser using natural language prompts.
*   **Easy Setup:** Get started quickly with a simple pip install.
*   **Cloud Version:** Skip setup with our [hosted version](https://cloud.browser-use.com) for instant browser automation.
*   **MCP Integration:** Support for [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) enabling integration with tools like Claude Desktop.
*   **Open Source:** Customize and extend the functionality to fit your needs.
*   **Extensive Examples:** Explore diverse use cases with provided examples.

## Quick Start

1.  **Installation:**

    ```bash
    pip install browser-use
    playwright install chromium --with-deps --no-shell
    ```
2.  **Basic Usage:**

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
3.  **API Keys:** Add your API keys for desired providers to a `.env` file:

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
4.  **CLI:** Test browser-use using its interactive CLI:

    ```bash
    pip install "browser-use[cli]"
    browser-use
    ```

For more details and advanced configurations, refer to the comprehensive [documentation 📕](https://docs.browser-use.com).

## Model Context Protocol (MCP) Integration

Browser Use seamlessly integrates with the Model Context Protocol (MCP), expanding capabilities through connections with other MCP servers.

### Use as MCP Server with Claude Desktop

Configure Claude Desktop to use Browser Use's automation features:

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

Integrate external MCP servers within your Browser Use agents to expand capabilities:

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

## Demos

Explore the power of Browser Use with these demonstrations:

*   **Shopping Automation:**  [Add groceries to a cart and checkout](https://www.youtube.com/watch?v=L2Ya9PYNns8).

    <br/>

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

    <br/>
*   **LinkedIn to Salesforce Integration:**  Automatically add new LinkedIn followers to Salesforce leads.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

    <br/>
*   **Job Application Automation:** Find and apply for jobs based on your resume.

    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

    <br/>
*   **Document Creation:** Write a letter in Google Docs and save it as a PDF.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

    <br/>
*   **Hugging Face Model Search:** Look up models with a specific license and save them to a file.

    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

    <br/>

## More Examples

Check out the  [examples](examples) folder or our [Discord](https://link.browser-use.com/discord) for more project inspiration. Also, find prompt ideas in our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo.

## Vision

Browser Use's vision is to allow users to tell their computer what to do, and have it done.

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

Contributions are welcome!  Please submit any bugs or feature requests as issues. To contribute to the documentation, visit the `/docs` folder.

## 🧪 Robust Agent Testing

Automate your agent validation with our CI:

*   **Add Tasks:**  Add a YAML file to `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automatic Validation:** Your task will automatically run and be evaluated against your criteria upon every update.

## Local Setup

For detailed setup instructions, consult the [local setup 📕](https://docs.browser-use.com/development/local-setup) documentation.

*   **Development Branch:** `main` is the active development branch.
*   **Production:** Install a stable [versioned release](https://github.com/browser-use/browser-use/releases) for production use.

---

## Swag

Show off your support with Browser-use swag! Browse our [Merch store](https://browsermerch.com). Contributors may receive complimentary swag.

## Citation

If you utilize Browser Use in your research or project, please cite:

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