<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo_black.svg">
    <img alt="mcp use logo" src="./static/logo-white.svg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<br>

# Supercharge Your Language Model with MCP-Use

🌐 **MCP-Use empowers you to connect any Large Language Model (LLM) to any Model Context Protocol (MCP) server**, unlocking access to tools and capabilities beyond simple text generation, all without vendor lock-in. Visit the [original repo](https://github.com/mcp-use/mcp-use) for the source code.

<p align="center">
    <a href="https://github.com/pietrozullo/mcp-use/stargazers" alt="GitHub stars">
        <img src="https://img.shields.io/github/stars/pietrozullo/mcp-use?style=social" /></a>
    <a href="https://pypi.org/project/mcp_use/" alt="PyPI Version">
        <img src="https://img.shields.io/pypi/v/mcp_use.svg"/></a>
    <a href="https://github.com/pietrozullo/mcp-use/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/pietrozullo/mcp-use" /></a>
    <a href="https://pypi.org/project/mcp_use/" alt="PyPI Downloads">
        <img src="https://static.pepy.tech/badge/mcp-use" /></a>
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
</div>

## Key Features

*   ✅ **LLM Agnostic:** Compatible with any LangChain-supported LLM that supports tool calling (e.g., OpenAI, Anthropic, Groq, Llama).
*   ✅ **Flexible Tool Access:** Easily connect your LLM to tools like web browsing, file operations, and more.
*   ✅ **Ease of Use:** Create MCP-capable agents with just a few lines of code.
*   ✅ **HTTP Support:** Connect to MCP servers running on specific HTTP ports.
*   ✅ **Dynamic Server Selection:** Agents can automatically choose the best MCP server for a task.
*   ✅ **Multi-Server Support:** Utilize multiple MCP servers simultaneously.
*   ✅ **Tool Access Control:** Restrict potentially dangerous tools for enhanced security.
*   ✅ **Custom Agent Building:** Integrate with any framework using the LangChain adapter or create your own.
*   ✅ **Sandboxed Execution:**  Run MCP servers securely in an isolated environment using E2B.
*   ✅ **Asynchronous Streaming:** Get real-time updates and progress with the `astream` method.

## Quick Start

### Installation

```bash
pip install mcp-use
```

### Installing LangChain Providers

You'll need to install the appropriate LangChain provider package for your chosen LLM.  For example:

```bash
pip install langchain-openai
```

Add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

> **Important**: Only models with tool calling capabilities can be used with mcp_use.

### Example Code

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

Explore the [mcp-use.com website](https://mcp-use.com/) to build and deploy agents, and visit the [mcp-use docs](https://docs.mcp-use.com/) for more in-depth information.

## Streaming Agent Output

Use `agent.astream(query)` to receive incremental results and real-time progress updates:

```python
async for chunk in agent.astream("Find the best restaurant in San Francisco"):
    print(chunk["messages"], end="", flush=True)
```

## Example Use Cases

*   **Web Browsing with Playwright:**  Easily integrate your LLM with web browsing capabilities.
*   **Airbnb Search:**  Build agents to search for accommodations with specific criteria.
*   **Blender 3D Creation:**  Generate 3D models using your LLM.
*   **HTTP Connection:** Connect to MCP servers running on HTTP ports.

## Additional Features

*   **Configuration File Support:** Manage MCP server setups with easy-to-use configuration files.
*   **Multi-Server Support:** Utilize multiple MCP servers simultaneously.
*   **Dynamic Server Selection (Server Manager):** Enhance efficiency by automatically selecting the right MCP server.
*   **Tool Access Control:** Restrict which tools are available to the agent for enhanced security.
*   **Custom Agent Building:** Adapt to your specific needs with the LangChain adapter.
*   **Debugging:** Utilize built-in debug mode for diagnosing issues.

## Detailed Documentation

For in-depth information, check out the:

*   [Website](https://mcp-use.com/)
*   [Documentation](https://docs.mcp-use.com/)

## Contributing

We welcome contributions! Please review the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines.

## Contributors

[Contributors Image](https://contrib.rocks/image?repo=mcp-use/mcp-use)

## Top Starred Dependents

[Top Starred Dependents Table]

## Requirements

*   Python 3.11+
*   MCP implementation (like Playwright MCP)
*   LangChain and appropriate model libraries (OpenAI, Anthropic, etc.)

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

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=732589b6-6850-4b8c-aa25-906c0979e426&page=README.md" />