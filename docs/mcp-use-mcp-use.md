<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<div align="center">
  <h2>🎉 <strong>We're LIVE on Product Hunt!</strong> 🎉</h2>
  <p><strong>Support us today and help us reach #1!</strong></p>
  <a href="https://www.producthunt.com/products/mcp-use?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-mcp&#0045;use" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1002629&theme=neutral&t=1754609432704" alt="mcp&#0045;use - Open&#0032;source&#0032;SDK&#0032;and&#0032;infra&#0032;for&#0032;MCP&#0032;servers&#0032;&#0038;&#0032;agents | Product Hunt" style="width: 220px; height: 54px;" width="250" height="54" /></a>
  <p>👆 <em>Click to upvote and leave a comment!</em></p>
</div>

<h1 align="center">🚀 **MCP-Use: Build Powerful Agents to Connect Any LLM to Any Tool with Open Source Flexibility**</h1>

<p align="center">
    <a href="https://github.com/pietrozullo/mcp-use/stargazers" alt="GitHub stars">
        <img src="https://img.shields.io/github/stars/pietrozullo/mcp-use?style=social" /></a>
    <a href="https://pypi.org/project/mcp_use/" alt="PyPI Downloads">
        <img src="https://static.pepy.tech/badge/mcp-use" /></a>
    <a href="https://pypi.org/project/mcp_use/" alt="PyPI Version">
        <img src="https://img.shields.io/pypi/v/mcp_use.svg"/></a>
    <a href="https://github.com/mcp-use/mcp-use-ts" alt="TypeScript">
      <img src="https://img.shields.io/badge/TypeScript-mcp--use-3178C6?logo=typescript&logoColor=white" /></a>
    <a href="https://github.com/pietrozullo/mcp-use/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/pietrozullo/mcp-use" /></a>
    <a href="https://docs.mcp-use.com" alt="Documentation">
        <img src="https://img.shields.io/badge/docs-mcp--use.com-blue" /></a>
    <a href="https://mcp-use.com" alt="Website">
        <img src="https://img.shields.io/badge/website-mcp--use.com-blue" /></a>
    </p>
    <p align="center">
    <a href="https://x.com/pietrozullo" alt="Twitter Follow - Pietro">
        <img src="https://img.shields.io/twitter/follow/Pietro?style=social" /></a>
    <a href="https://x.com/pederzh" alt="Twitter Follow - Luigi">
        <img src="https://img.shields.io/twitter/follow/Luigi?style=social" /></a>
    <a href="https://discord.gg/XkNkSkMz3V" alt="Discord">
        <img src="https://dcbadge.limes.pink/api/server/XkNkSkMz3V?style=flat" /></a>
</p>

[**Get started now and explore the documentation!**](https://mcp-use.com/)

---

## What is MCP-Use?

**MCP-Use** is an open-source Python library that empowers developers to seamlessly integrate any Large Language Model (LLM) with any Model Context Protocol (MCP) server. Build custom agents with robust tool access without relying on closed-source solutions or proprietary application clients.

## Key Features

*   **Flexibility & Integration**: Connect any LLM, including those supported by LangChain (OpenAI, Anthropic, Groq, etc.), to MCP servers.
*   **Ease of Use:** Quickly create an MCP-capable agent with just a few lines of code.
*   **Code Builder**: Explore MCP capabilities and generate starter code with the interactive [code builder](https://mcp-use.com/builder).
*   **HTTP Support**: Directly connect to MCP servers through HTTP, including those running on specific ports.
*   **Dynamic Server Selection (Server Manager)**: Agents intelligently select the best MCP server for each task from a pool of available servers.
*   **Multi-Server Support**: Use multiple MCP servers simultaneously within a single agent for complex workflows.
*   **Tool Access Control**: Restrict tool access for enhanced security and control.
*   **Custom Agents**: Build bespoke agents using the LangChain adapter or create new adapters.
*   **Streaming Output**: Real-time feedback and incremental results with asynchronous streaming.
*   **Sandboxed Execution (E2B)**: Run MCP servers in a secure, isolated environment using E2B's cloud infrastructure.
*   **Direct Tool Calls:** Programmatically call MCP server tools without relying on an LLM.

## Quick Start

### Installation

Install MCP-Use using `pip`:

```bash
pip install mcp-use
```

Or install from source:

```bash
git clone https://github.com/mcp-use/mcp-use.git
cd mcp-use
pip install -e .
```

### Dependencies
You will need to install dependencies for your chosen LLM and other tools:
*  Install the LangChain provider you want to use with `pip install langchain-openai` or `pip install langchain-anthropic`.
*  Install any additional libraries used by the MCP server, such as `npx @playwright/mcp@latest` for the Playwright web browser.
*  Add your API keys for the provider you want to use to your `.env` file
*   *Important:* Only models with tool calling capabilities can be used with mcp_use.

### Code Example

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary
    config = {
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": ["@playwright/mcp@latest"],
          "env": {
            "DISPLAY": ":1"
          }
        }
      }
    }

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(config)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "Find the best restaurant in San Francisco",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

For other settings, models, and more, check out the documentation.

## Example Use Cases

*   **Web Browsing with Playwright:**  See the example above, but remember to create a browser_mcp.json file, and install the playwright dependencies
*   **Airbnb Search:** Utilizes the airbnb_mcp.json configuration file, and the proper dependencies.
*   **Blender 3D Creation:**  Uses the Blender setup, config, and LLM settings to create an agent that creates 3D models.

## [View Full Example Code and Configuration Files](https://github.com/mcp-use/mcp-use)

---

## Advanced Features

*   **Configuration Support**: Configure your MCP server connections through various methods, including JSON files and environment variables.
*   **Multi-Server Support**: Connect to and manage multiple MCP servers simultaneously.
*   **Dynamic Server Selection (Server Manager)**: Optimize agent performance by intelligently selecting the appropriate server.
*   **Tool Access Control**:  Enhance security by restricting which tools are available to the agent.
*   **Sandboxed Execution**: Run MCP servers in a secure, isolated environment (requires E2B API key)
*   **Direct Tool Calls**: Call tools directly.
*   **Build a Custom Agent**: Create custom agent using the LangChain adapter.

## Contributing

We welcome contributions!  Check out our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Key Dependencies

*   Python 3.11+
*   LangChain
*   MCP server implementation (e.g., Playwright MCP)
*   Appropriate LLM provider libraries (e.g., `langchain-openai`, `langchain-anthropic`)

## License

MIT

## Citation

```bibtex
@software{mcp_use2025,
  author = {Zullo, Pietro},
  title = {MCP-Use: MCP Library for Python},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pietrozullo/mcp-use}
}
```

## Get Involved

*   [Explore the Documentation](https://mcp-use.com/)
*   [Visit the mcp-use-ts (TypeScript version)](https://github.com/mcp-use/mcp-use-ts)
*   [Contribute on GitHub](https://github.com/mcp-use/mcp-use)
*   [Join the Discord](https://discord.gg/XkNkSkMz3V)
*   [Follow on Twitter](https://twitter.com/pietrozullo)

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=732589b6-6850-4b8c-aa25-906c0979e426&page=README.md" />