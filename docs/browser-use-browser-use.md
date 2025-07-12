<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI</h1>

<p align="center">
  <b>Unleash the power of AI to control your browser, automating tasks and streamlining your workflow!</b>
  <br>
  <a href="https://github.com/browser-use/browser-use">
    <img src="https://img.shields.io/github/stars/browser-use/browser-use?style=social" alt="GitHub stars">
  </a>
  <a href="https://link.browser-use.com/discord">
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
</p>

## Key Features

*   **AI-Powered Browser Automation:** Control your browser with natural language.
*   **Easy Setup:** Get started quickly with simple installation steps.
*   **Cloud Integration:** Try our hosted version for instant automation.
*   **MCP Support:** Integrate with the Model Context Protocol for extended capabilities.
*   **Interactive CLI:** Test and experiment with the browser-use interactive CLI.
*   **Extensive Documentation:** Comprehensive documentation to guide you.
*   **Robust Testing:** Automated task validation to ensure reliability.
*   **Community Support:** Join our vibrant [Discord](https://link.browser-use.com/discord) community.

## Quick Start

Install Browser Use using pip:

```bash
pip install browser-use
```

Install a browser (e.g., Chromium):

```bash
playwright install chromium --with-deps --no-shell
```

Example Python code to start an agent:

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

Add your API keys to your `.env` file:

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

For detailed configuration and advanced usage, consult the [documentation](https://docs.browser-use.com).

### Test with UI

Test browser-use using its [Web UI](https://github.com/browser-use/web-ui) or [Desktop App](https://github.com/browser-use/desktop).

### Test with an interactive CLI

You can also use our `browser-use` interactive CLI:

```bash
pip install "browser-use[cli]"
browser-use
```

## MCP Integration

Browser-use seamlessly integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), enabling powerful extensions and integrations.

### Use as MCP Server with Claude Desktop

Integrate browser-use within your Claude Desktop configuration:

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

Extend your agent's capabilities by connecting to multiple MCP servers:

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

Learn more in the [MCP documentation](https://docs.browser-use.com/customize/mcp-server).

## Demos

Explore real-world use cases:

<br/><br/>

**1. Grocery Shopping Automation**

[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

**2. LinkedIn to Salesforce Integration**

Prompt: Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

**3. Job Application Automation**

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

**4. Document Generation**

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

**5. Data Extraction and Sorting**

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## Explore More

Find more examples in the [examples](examples) folder or share your projects in our [Discord](https://link.browser-use.com/discord).  Also, check out our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompt inspiration.

## Vision

Empowering your computer to perform tasks through natural language commands.

## Roadmap

### Agent

*   \[ ] Improve agent memory to handle +100 steps
*   \[ ] Enhance planning capabilities (load website specific context)
*   \[ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

*   \[ ] Enable detection for all possible UI elements
*   \[ ] Improve state representation for UI elements so that all LLMs can understand what's on the page

### Workflows

*   \[ ] Let user record a workflow - which we can rerun with browser-use as a fallback
*   \[ ] Make rerunning of workflows work, even if pages change

### User Experience

*   \[ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
*   \[ ] Improve docs
*   \[ ] Make it faster

### Parallelization

*   \[ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.

## Contributing

We welcome contributions!  Report bugs and request features via issues. Contribute to the docs in the `/docs` folder.

## 🧪 Robust Task Validation

Ensure task reliability with automated CI testing.

*   **Submit your task:**  Add a YAML file to `tests/agent_tasks/` (see the [`README`](tests/agent_tasks/README.md) for details).
*   **Automatic validation:**  Your task will be run and evaluated on every update.

## Local Setup

Learn more in the [local setup 📕](https://docs.browser-use.com/development/local-setup).

**Note:** `main` is the primary development branch. For production, use a [versioned release](https://github.com/browser-use/browser-use/releases).

---

## Swag

Show off your Browser-use swag! Check out our [Merch store](https://browsermerch.com). Good contributors may receive swag!

## Citation

If you use Browser Use, please cite it:

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

  <a href="https://x.com/intent/user?screen_name=gregpr07">
    <img src="https://img.shields.io/twitter/follow/Gregor?style=social" alt="Twitter Follow">
  </a>
  <a href="https://x.com/intent/user?screen_name=mamagnus00">
    <img src="https://img.shields.io/twitter/follow/Magnus?style=social" alt="Twitter Follow">
  </a>
</div>

<div align="center">
  Made with ❤️ in Zurich and San Francisco
</div>
```

Key improvements:

*   **SEO Optimization:** Added relevant keywords like "AI," "browser automation," "automation," and "natural language" throughout the document.
*   **Clear Structure:** Uses headings and bullet points for readability and scannability.
*   **Concise Summary:**  Summarizes the key features for quick understanding.
*   **Compelling Hook:** The introductory sentence immediately grabs the reader's attention.
*   **Call to Action:** Encourages readers to try the cloud version and explore the community.
*   **Emphasis:** Highlights key features with bold text.
*   **Complete:**  Includes all essential information from the original README and expands on certain sections.
*   **Improved Formatting:** Consistent use of Markdown for better presentation.
*   **Conciseness:** Removes some redundancies while keeping crucial information.
*   **Enhanced Demos:**  Provides a clear description for each demo and includes the prompts.
*   **Contributors:** Offers clear contribution guidelines.
*   **Roadmap:** The "Roadmap" section is now within a clearly defined section.
*   **Local Setup:** Provides a direct link to the local setup documentation.
*   **Emphasis on Community and Support:** Encourages interaction and contribution to the project's growth.