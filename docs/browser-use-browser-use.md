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

**Browser Use empowers you to control your web browser with the power of AI, enabling automation and unlocking new possibilities for web interaction.**

## Key Features

*   **AI-Powered Automation:** Leverage AI to automate complex browser tasks.
*   **Easy Setup:**  Get started quickly with a simple Python installation.
*   **Cloud Option:**  Utilize the [Browser Use Cloud](https://cloud.browser-use.com) for fast, scalable, and stealth-enabled automation.
*   **Model Context Protocol (MCP) Integration:** Compatible with MCP for integration with Claude Desktop and other MCP-compatible clients.
*   **Extensive Examples:**  Explore diverse use cases and prompts to spark your creativity.
*   **Robust Testing:**  Ensure your agents' reliability with CI integration for task validation.
*   **Customizable Workflows:**  Create and run automated workflows.
*   **Active Community:**  Join the [Discord](https://link.browser-use.com/discord) and learn from other developers.

## Quickstart

Install Browser Use using pip:

```bash
pip install browser-use
```

**Note**: If you do not have Chrome or Chromium installed, use the following command to download the latest Chromium:

```bash
uvx playwright install chromium --with-deps --no-shell
```

Spin up your agent:

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

Add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
```

For more details, explore the official [Documentation 📕](https://docs.browser-use.com).

## Demos

Here are a few examples to get you started:

*   **Grocery Shopping:** [Add grocery items to cart, and checkout](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py)

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
*   **LinkedIn to Salesforce Integration:** Add your latest LinkedIn follower to your leads in Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
*   **Job Application Automation:**  [Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py).

    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
*   **Letter Writing:** [Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py).

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
*   **Hugging Face Model Search:**  [Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py).

    https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

## Explore More Examples

Find more examples in the [examples](examples) folder and explore the `awesome-prompts` repo [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) for prompting inspiration.

## Model Context Protocol (MCP) Integration

Browser-use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing integration with Claude Desktop and other MCP-compatible clients.

### Using as an MCP Server with Claude Desktop

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

This configuration gives Claude Desktop access to browser automation tools, which opens up the ability for web scraping, form filling, and more.

### Connecting External MCP Servers to Browser-Use Agent

Browser-use agents can connect to multiple external MCP servers to extend their capabilities:

```python
import asyncio
from browser_use import Agent, Controller, ChatOpenAI
from browser_use.mcp.client import MCPClient

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
        task="Find the latest pdf report in my documents and create a GitHub issue about it",
        llm=ChatOpenAI(model="gpt-4.1-mini"),
        controller=controller  # Controller has tools from both MCP servers
    )

    # Run the agent
    await agent.run()

    # Cleanup
    await filesystem_client.disconnect()
    await github_client.disconnect()

asyncio.run(main())
```

Check the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

## Vision

Automate your browser tasks with the help of AI.

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

We welcome your contributions! Feel free to open issues for bug reports or feature requests. To contribute to the docs, look in the `/docs` folder.

## 🧪 Robust Agent Validation

Ensure task reliability through CI integration and automatic validation.

*   **Add Your Task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automated Validation:** Your task is automatically run and evaluated upon every update.

## Local Setup

Explore the library further through [local setup 📕](https://docs.browser-use.com/development/local-setup).

The `main` branch is under active development. For production purposes, install a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Visit our [Merch store](https://browsermerch.com).

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

Key improvements and SEO considerations:

*   **Clear, Concise Hook:** The one-sentence hook is now at the very beginning and is very descriptive.
*   **Keyword Optimization:** The title and key headings incorporate relevant keywords like "AI," "automation," and "browser."
*   **Structured Content:** The README is better organized using headings, subheadings, and bullet points to improve readability and SEO.
*   **Detailed Descriptions:** The descriptions under each heading are more informative.
*   **Calls to Action:** The text includes calls to action.
*   **Internal Linking:** The text uses internal links to connect different parts of the README.
*   **Concise language:** Uses concise language to deliver all the information.
*   **Removed Redundancy** Removed some redundancy in the document for better flow.
*   **Focus on Benefits:** Highlights the benefits of the project, which is critical for attracting users and improving SEO.
*   **Alt Text:** Added alternative text to the images to help with SEO.
*   **Clear Installation Instructions:** The install and quickstart are concise.
*   **Markdown Format:** The markdown format is clean and readable.
*   **Citation:** Added the citation for the project to help credit the authors.
*   **Emphasis on community:** Helps foster a community around the project.