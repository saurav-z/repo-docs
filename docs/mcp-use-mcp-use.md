<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo_black.svg">
    <img alt="mcp use logo" src="./static/logo-white.svg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<br>

# MCP-Use: Connect Any LLM to Any MCP Server 🚀

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

🌐 **MCP-Use empowers developers to seamlessly integrate any Large Language Model (LLM) with any MCP server**, opening the door to building powerful, custom agents with tool access. This allows you to leverage the capabilities of tools like web browsing, file operations, and more without relying on closed-source solutions.

**Key Features:**

*   ✅ **LLM Agnostic**: Works with all LangChain-supported LLMs that offer tool calling capabilities (OpenAI, Anthropic, Groq, Llama, etc.).
*   💻 **Rapid Development**: Build your first MCP-enabled agent with just a few lines of code.
*   🌐 **Interactive Code Builder**: Generate starter code and explore MCP functionality using the <a href="https://mcp-use.com/builder">interactive code builder</a>.
*   🔗 **HTTP Integration**: Connect to MCP servers running on specific HTTP ports.
*   ⚙️ **Dynamic Server Selection**: Intelligent agent behavior that dynamically selects the best MCP server based on task requirements.
*   🧩 **Multi-Server Support**: Leverage multiple MCP servers simultaneously within a single agent.
*   🛡️ **Tool Access Control**: Restrict potentially harmful tools (file system, network access) for enhanced security.
*   🔧 **Custom Agent Creation**: Build your own agents using your preferred framework, leveraging our LangChain adapter or creating new adapters.
*   ✨ **Streaming Output**: Receive real-time updates with the `astream` method, supporting real-time feedback.
*   ☁️ **Sandboxed Execution**: Run MCP servers securely using E2B's cloud infrastructure.
*   🔎 **Debugging**: Easily diagnose issues in your agent implementation.

[Find out more at the original repo!](https://github.com/mcp-use/mcp-use)

## Quick Start

**Installation:**

```bash
pip install mcp-use
```

Or install from source:

```bash
git clone https://github.com/pietrozullo/mcp-use.git
cd mcp-use
pip install -e .
```

**Install LangChain Providers:**

Install the appropriate LangChain provider package for your chosen LLM. For example:

```bash
# For OpenAI
pip install langchain-openai

# For Anthropic
pip install langchain-anthropic
```

Add your API keys to your `.env` file for the provider you want to use.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

> **Important**:  Make sure your chosen model supports function calling or tool use.

**Example Usage:**

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

## Streaming Agent Output

Use the `astream` method on `MCPAgent` for asynchronous streaming of agent output:

```python
async for chunk in agent.astream("Find the best restaurant in San Francisco"):
    print(chunk["messages"], end="", flush=True)
```

## Example Use Cases

**Web Browsing with Playwright:**

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()

    # Create MCPClient from config file
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "browser_mcp.json")
    )

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")
    # Alternative models:
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    # llm = ChatGroq(model="llama3-8b-8192")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "Find the best restaurant in San Francisco USING GOOGLE SEARCH",
        max_steps=30,
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Airbnb Search:** (See the original README for details.)

**Blender 3D Creation:** (See the original README for details.)

## Configuration

**HTTP Connection Example:** (See the original README for details.)

**Multi-Server Support:** (See the original README for details.)

**Dynamic Server Selection (Server Manager):** (See the original README for details.)

## Tool Access Control

(See the original README for details.)

## Sandboxed Execution

(See the original README for details.)

## Build a Custom Agent:

(See the original README for details.)

## Debugging

(See the original README for details.)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contributors

(See the original README for details.)

## Top Starred Dependents

<!-- gh-dependents-info-used-by-start -->

(See the original README for details.)

<!-- gh-dependents-info-used-by-end -->

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