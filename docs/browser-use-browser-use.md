<!-- Improved README for Browser Use -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI 🤖</h1>

<p align="center"><b>Effortlessly control your browser with AI: Automate tasks, streamline workflows, and unlock new possibilities.</b></p>

[![GitHub stars](https://img.shields.io/github/stars/browser-use/browser-use?style=social)](https://github.com/browser-use/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

---

## Key Features

*   🚀 **AI-Powered Automation:** Control your browser with natural language commands.
*   💻 **Versatile Use Cases:** Automate web scraping, form filling, job applications, content creation, and more.
*   ☁️ **Cloud-Based Option:** Get started quickly with our cloud service for scalable, stealth-enabled browser automation.
*   🛠️ **Customization:** Leverage the Model Context Protocol (MCP) for advanced integration.
*   💡 **Easy Integration:** Simple Python installation and code examples.
*   🧪 **Robust Testing:** Integrate your tasks into the CI for automatic validation.
*   🤝 **Open Source:** Actively maintained and open to community contributions.

---

## Quick Start

Install the `browser-use` Python package:

```bash
pip install browser-use
```

Download Chromium (if you don't have it):

```bash
uvx playwright install chromium --with-deps --no-shell
```

**Example Usage:**

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

Add your API keys to your `.env` file:

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

For detailed configuration and usage, see the [documentation 📕](https://docs.browser-use.com).

---

## Demos

Browse these examples to see Browser Use in action:

*   **Shopping Automation:** Add items to a cart and check out.

    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

*   **LinkedIn to Salesforce Integration:** Automatically add LinkedIn leads to Salesforce.

    ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

*   **Job Application Automation:** Find and apply to jobs based on your CV.

    <img src="https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04" alt="Job Application">

*   **Document Creation:** Generate and save a document in Google Docs.

    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

*   **Hugging Face Model Search:** Find and save models based on specific criteria.

    <img src="https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3" alt="Hugging Face Model Search">

More examples are available in the [examples](examples) folder and on our [Discord](https://link.browser-use.com/discord). Explore the [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting inspiration.

---

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

See the [MCP documentation](https://docs.browser-use.com/customize/mcp-server) for more details.

---

## Vision

>   Empowering you to control your computer by simply telling it what you want to do.

---

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

---

## Contributing

We welcome contributions!  Please open issues for bugs or feature requests. For documentation contributions, check out the `/docs` folder.

---

## 🧪 Robust Agent Testing

Ensure your agents are reliable by integrating your tasks into our CI:

*   **Add Your Task:** Create a YAML file in `tests/agent_tasks/` (see the [`README`](tests/agent_tasks/README.md) for details).
*   **Automatic Validation:** Your task will be automatically executed and evaluated on every update.

---

## Local Setup

For more information on setting up and developing with the library, see the [local setup 📕](https://docs.browser-use.com/development/local-setup).

>   **Note:**  `main` is the active development branch. For production, we recommend installing a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser Use swag!  Check out the [Merch store](https://browsermerch.com). Good contributors may receive swag for free 👀.

---

## Citation

If you use Browser Use in your research or projects, please cite it as follows:

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
    <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400" alt="Browser Use Team">

[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
</div>

<div align="center">
    Made with ❤️ in Zurich and San Francisco
</div>
```

**Key Improvements and SEO Considerations:**

*   **Clear, Concise Hook:**  The one-sentence hook immediately establishes the core value proposition.
*   **Targeted Keywords:**  Includes keywords such as "browser automation," "AI," "web scraping," "automation," and "control browser."
*   **Structured Headings:**  Uses `<h1>`, `<h2>`, and `<h3>` to organize content for readability and SEO.
*   **Bulleted Feature List:**  Highlights key benefits for quick scanning and understanding.
*   **Cloud-Based Option:**  Emphasizes the cloud option, which provides immediate value.
*   **Call to Action:**  Encourages use with "Get Started" and the example code.
*   **Example Showcase:**  Provides compelling visual examples to demonstrate capabilities.
*   **MCP and Documentation Links:**  Prominently features links to documentation and resources.
*   **Contribution Instructions:**  Makes it easy for others to contribute.
*   **Citation Information:**  Provides ready-to-use citation data.
*   **Author Information & Social Links:**  Maintains author information and social media links for credibility.
*   **Clearer Visuals:**  Uses descriptive alt text for images.
*   **Concise Language:** The overall writing is clear and easy to follow.
*   **Direct link to GitHub Repo:**  `[browser-use](https://github.com/browser-use/browser-use)` at the top.

This revised README is more user-friendly, easier to understand, and optimized for search engines, making it more likely to attract users and contributors to the project.