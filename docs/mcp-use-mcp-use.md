<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<h1 align="center">🚀 Connect LLMs to Tools with MCP-Use</h1>

<p align="center">
    <a href="https://github.com/mcp-use/mcp-use/stargazers" alt="GitHub stars">
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
</div>

🌐 **MCP-Use empowers developers to effortlessly connect any Language Learning Model (LLM) to any Model Context Protocol (MCP) server, enabling the creation of custom agents with tool access.** Build agents that interact with web browsers, file systems, and more, all without relying on closed-source solutions. Dive into our [original repo](https://github.com/mcp-use/mcp-use) to get started.

**Key Features:**

*   ✅ **Ease of Use:** Get started in just six lines of code.
*   🤖 **LLM Flexibility:** Works with any LangChain-supported LLM that supports tool calling (e.g., OpenAI, Anthropic, Groq).
*   🌐 **Code Builder:** Explore MCP capabilities and generate starter code with the interactive [code builder](https://mcp-use.com/builder).
*   🔗 **HTTP Support:** Connect directly to MCP servers running on specific HTTP ports.
*   ⚙️ **Dynamic Server Selection:** Agents intelligently choose the best MCP server for each task using the Server Manager.
*   🧩 **Multi-Server Support:** Utilize multiple MCP servers simultaneously within a single agent.
*   🛡️ **Tool Restrictions:** Implement tool access control for enhanced security.
*   🔧 **Custom Agents:** Build your own custom agents with flexible adapters.
*   ❓ **Community Driven:** [What should we build next?](https://mcp-use.com/what-should-we-build-next)

**Installation:**

```bash
pip install mcp-use
```

or

```bash
git clone https://github.com/mcp-use/mcp-use.git
cd mcp-use
pip install -e .
```

**Example: Web Browsing with Playwright**

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

**See also:**

*   [mcp-use.com website](https://mcp-use.com/) to build and deploy agents with your favorite MCP servers.
*   [mcp-use docs](https://docs.mcp-use.com/) to get started with mcp-use library
*   [mcp-use-ts](https://github.com/mcp-use/mcp-use-ts) for the TypeScript version

**And more:**

*   [Streaming Agent Output](#streaming-agent-output)
*   [Example Use Cases](#example-use-cases)
*   [Configuration Support](#configuration-support)
*   [Multi-Server Support](#multi-server-support)
*   [Dynamic Server Selection (Server Manager)](#dynamic-server-selection-server-manager)
*   [Tool Access Control](#tool-access-control)
*   [Sandboxed Execution](#sandboxed-execution)
*   [Direct Tool Calls (Without LLM)](#direct-tool-calls-without-llm)
*   [Build a Custom Agent](#build-a-custom-agent)
*   [Debugging](#debugging)
*   [Contributors](#contributors)
*   [Top Starred Dependents](#top-starred-dependents)
*   [Requirements](#requirements)
*   [License](#license)
*   [Citation](#citation)

---
**Contribute!**
We welcome your contributions!  Check out our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines for details.
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=732589b6-6850-4b8c-aa25-906c0979e426&page=README.md" />