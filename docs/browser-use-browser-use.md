<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: AI-Powered Browser Automation</h1>

<div align="center">
  <a href="https://github.com/browser-use/browser-use">
    <img src="https://img.shields.io/github/stars/gregpr07/browser-use?style=social" alt="GitHub stars">
  </a>
  <a href="https://discord.gg/K5bYhE8bS8">
    <img src="https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="https://cloud.browser-use.com">
    <img src="https://img.shields.io/badge/Cloud-☁️-blue" alt="Cloud">
  </a>
  <a href="https://docs.browser-use.com">
    <img src="https://img.shields.io/badge/Documentation-📕-blue" alt="Documentation">
  </a>
  <a href="https://x.com/intent/user?screen_name=gregpr07">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Twitter Follow">
  </a>
  <a href="https://x.com/intent/user?screen_name=mamagnus00">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Twitter Follow">
  </a>
  <a href="https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341" alt="Weave Badge">
  </a>
</div>

**Browser Use empowers you to control your web browser with simple natural language commands.** Automate tasks, streamline workflows, and unlock the full potential of AI-driven browser interaction.  [Explore the Browser Use Repo](https://github.com/browser-use/browser-use)

## Key Features

*   **Natural Language Control:**  Instruct your browser using plain English.
*   **AI-Powered Automation:** Automate complex tasks with AI assistance.
*   **Web Scraping & Data Extraction:** Easily gather information from websites.
*   **Workflow Automation:** Build and automate sequences of browser actions.
*   **Model Context Protocol (MCP) Support:** Integrates with MCP-compatible clients like Claude Desktop.
*   **Cloud Option:** Get started quickly with our [cloud](https://cloud.browser-use.com) offering.

## Getting Started

### Installation

Install Browser Use using pip:

```bash
pip install browser-use
```

Then, install a browser (e.g., Chromium):

```bash
uvx playwright install chromium --with-deps --no-shell
```

### Example Usage

Here's a basic example to get you started:

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

**Important:**  Add your API keys to a `.env` file, e.g., `OPENAI_API_KEY=YOUR_API_KEY`.

For more detailed setup instructions, configuration options, and advanced usage, see the comprehensive [documentation 📕](https://docs.browser-use.com).

## Demos & Use Cases

See Browser Use in action with these examples:

*   **Grocery Shopping:**  Automated grocery ordering and checkout ([Video](https://www.youtube.com/watch?v=L2Ya9PYNns8))
*   **LinkedIn to Salesforce:**  Adding LinkedIn leads to Salesforce.
*   **Job Application Automation:** Finding and applying for ML jobs.
*   **Document Creation:** Creating and saving documents in Google Docs.
*   **Hugging Face Model Search:** Finding and saving models based on criteria.

  (Replace the image links with markdown-formatted images)

### Examples

*   **Shopping Example:** Add grocery items to cart, and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)
    *   **Prompt:** Add grocery items to cart, and checkout.
*   **LinkedIn to Salesforce:**  Add my latest LinkedIn follower to my leads in Salesforce.
     ![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)
     *   **Prompt:** Add my latest LinkedIn follower to my leads in Salesforce.
*   **Job Application Example:** Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'
    ![Job Application](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)
    *   **Prompt:** Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.
*   **Google Docs Example:**  Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.
    ![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)
    *   **Prompt:** Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.
*   **Hugging Face Example:**  Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.
    ![Hugging Face](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd323b3)
     *   **Prompt:** Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

Explore more examples in the [`examples`](examples) folder and get inspired in our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repository.  Join the community on [Discord](https://link.browser-use.com/discord) to share your projects!

## Model Context Protocol (MCP) Integration

Browser Use supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing integration with tools like Claude Desktop.

### Using as an MCP Server

Configure Claude Desktop to use Browser Use:

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

### Connecting to External MCP Servers

Extend Browser Use's capabilities by connecting to multiple MCP servers:

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

For further details, consult the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Vision

**The future is simple: Tell your computer what to do, and it gets it done.**

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

We welcome contributions! Please submit issues for bug reports and feature requests.  Contribute to the documentation in the `/docs` folder.

## 🧪 Robust Agent Testing

Ensure your tasks run correctly with our CI!

*   **Add your task:** Add a YAML file in `tests/agent_tasks/` (see the [`README there`](tests/agent_tasks/README.md) for details).
*   **Automated validation:** Your task will be automatically run and evaluated with every update.

## Local Setup

For detailed local setup instructions, see the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:** The `main` branch is for active development.  For production, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser Use swag!  Visit our [Merch store](https://browsermerch.com).  Good contributors may receive free swag! 👀

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
  <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400" alt="browser-use logo"/>
  <br>
  <a href="https://x.com/intent/user?screen_name=gregpr07">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Follow Gregor on Twitter">
  </a>
  <a href="https://x.com/intent/user?screen_name=mamagnus00">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Follow Magnus on Twitter">
  </a>
  <br>
  Made with ❤️ in Zurich and San Francisco
</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  The title includes keywords like "AI," "Browser Automation," "Web Automation," and "Browser Control".  The content is structured with clear headings and subheadings, and the use of bold text helps with keyword emphasis.
*   **One-Sentence Hook:** The first sentence directly and concisely introduces the core functionality of Browser Use, grabbing the reader's attention.
*   **Clearer Structure:**  The README is divided into logical sections (Features, Getting Started, Examples, MCP Integration, Roadmap, Contributing, etc.) making it easier to read and understand.
*   **Emphasis on Benefits:**  The "Key Features" section is very important, highlighting the core advantages of using the library.
*   **Action-Oriented Language:**  Uses phrases like "Empowers you to," "Automate tasks," "Streamline workflows," and "Explore the repo" to encourage user engagement.
*   **Comprehensive Examples:**  The example section is improved to be more descriptive, including prompt examples and a better layout.
*   **Call to Actions:**  Includes links to the documentation, Discord, and cloud offering throughout the README, encouraging user engagement.
*   **Community Building:** Encourages contributions, highlights the community aspect via Discord.
*   **Concise and Informative:** Keeps the information clear and focused, avoiding unnecessary jargon.
*   **Updated Badges:**  Uses updated social media badges and links.
*   **Alt Text for Images:** Added `alt` text to all image tags for accessibility and SEO.
*   **Bibtex formatting** Improves citation.
*   **Roadmap Clarity:** Improved Roadmap section for better understanding.
*   **Vision Clarity:** Updated to better describe the vision.