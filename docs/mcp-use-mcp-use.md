<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<div align="center">
  <h2>🎉 <strong>We're LIVE on Product Hunt!</strong> 🎉</h2>
  <p><strong>Support us today and help us reach #1!</strong></p>
  <a href="https://www.producthunt.com/products/mcp-use?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-mcp&#0045;use" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1002629&theme=neutral&t=1754609432704" alt="mcp&#0045;use - Open&#0032;source&#0032;SDK&#0032;and&#0032;infra&#0032;for&#0032;MCP&#0032;servers&#0038;&#0032;agents | Product Hunt" style="width: 220px; height: 54px;" width="250" height="54" /></a>
  <p>👆 <em>Click to upvote and leave a comment!</em></p>
</div>
</div>

# MCP-Use: Unlock the Power of LLMs with Open-Source MCP Agents

🌐 **MCP-Use** empowers developers to effortlessly connect any Large Language Model (LLM) to any **Model Context Protocol (MCP)** server, enabling the creation of versatile, custom agents with tool access, all within an open-source framework.  [Check out the original repo](https://github.com/mcp-use/mcp-use)!

<div align="center">
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
    <a href="https://x.com/pietrozullo" alt="Twitter Follow - Pietro">
        <img src="https://img.shields.io/twitter/follow/Pietro?style=social" /></a>
    <a href="https://x.com/pederzh" alt="Twitter Follow - Luigi">
        <img src="https://img.shields.io/twitter/follow/Luigi?style=social" /></a>
    <a href="https://discord.gg/XkNkSkMz3V" alt="Discord">
        <img src="https://dcbadge.limes.pink/api/server/XkNkSkMz3V?style=flat" /></a>
</div>

## Key Features

*   ✅ **Easy Integration:** Connect any LLM (OpenAI, Anthropic, etc.) to your preferred MCP servers.
*   🛠️ **Tool Access:** Empower your agents with access to web browsing, file operations, and more.
*   💻 **Code Builder:** Quickly generate starter code with our interactive [code builder](https://mcp-use.com/builder).
*   🌐 **HTTP Support:**  Connect directly to MCP servers via HTTP.
*   🔄 **Dynamic Server Selection:** Agents can dynamically choose the best server for the job.
*   📡 **Multi-Server Support:** Utilize multiple MCP servers simultaneously.
*   🔒 **Tool Restrictions:** Enhance security with tool access control.
*   🧩 **Custom Agents:** Build custom agents using the LangChain adapter or create new adapters.
*   🚀 **Sandboxed Execution:** Run MCP servers in a secure, isolated environment using E2B's infrastructure.
*   ⚡ **Streaming Output:**  Receive real-time updates and progress reports with asynchronous streaming.

## Getting Started

### Installation

Install `mcp-use` using pip:

```bash
pip install mcp-use
```

For source installation:

```bash
git clone https://github.com/pietrozullo/mcp-use.git
cd mcp-use
pip install -e .
```

### Install LangChain Providers

Install the necessary LangChain provider for your chosen LLM.  For example:

```bash
pip install langchain-openai  # For OpenAI
pip install langchain-anthropic # For Anthropic
```

Set your API keys in your `.env` file:

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
```

**Important:** Only models with tool/function calling capabilities are compatible.

### Basic Usage

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    load_dotenv()

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

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    result = await agent.run(
        "Find the best restaurant in San Francisco",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Examples

### Web Browsing with Playwright

```python
# Example usage (See full examples in README)
```

### Airbnb Search

```python
# Example usage (See full examples in README)
```

### Blender 3D Creation

```python
# Example usage (See full examples in README)
```

## Configuration

### HTTP Connection Example

```python
# Example usage (See full examples in README)
```

## Multi-Server Support

Configure and connect to multiple MCP servers simultaneously:

```json
{
  "mcpServers": {
    "airbnb": {
      "command": "npx",
      "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "env": {
        "DISPLAY": ":1"
      }
    }
  }
}
```

### Dynamic Server Selection (Server Manager)

```python
# Example usage (See full examples in README)
```

## Tool Access Control

Restrict tool access for enhanced security:

```python
# Example usage (See full examples in README)
```

## Sandboxed Execution

Run MCP servers in a secure sandbox:

```python
# Example usage (See full examples in README)
```

## Direct Tool Calls (Without LLM)

Call MCP server tools directly:

```python
# Example usage (See full examples in README)
```

## Build a Custom Agent

Build your own custom agent with LangChain Adapter:

```python
# Example usage (See full examples in README)
```

## Debugging

Enable debug mode to diagnose issues:

```bash
DEBUG=1 python3.11 your_script.py  # INFO level
DEBUG=2 python3.11 your_script.py  # DEBUG level
```
or programmatically

```python
import mcp_use
mcp_use.set_debug(1)  # INFO level
# or
mcp_use.set_debug(2)  # DEBUG level (full verbose output)
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contributors

[Link to contributors graph]

## Top Starred Dependents

[Link to top starred dependents table]

## Requirements

*   Python 3.11+
*   MCP Implementation (e.g., Playwright MCP)
*   LangChain and LLM libraries (e.g., OpenAI, Anthropic)

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