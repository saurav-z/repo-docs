<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>
</div>

## 🚀 Supercharge Your LLMs with MCP-Use: Connect, Build, and Automate!

[<img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1002629&theme=neutral&t=1754609432704" alt="mcp&#0045;use - Open&#0032;source&#0032;SDK&#0032;and&#0032;infra&#0032;for&#0032;MCP&#0032;servers&#0038;&#0032;agents | Product Hunt" style="width: 220px; height: 54px;" width="250" height="54" />](https://www.producthunt.com/products/mcp-use?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-mcp&#0045;use) 
<p>👆 <em>Click to upvote and leave a comment!</em></p>

<p align="center">
    <a href="https://github.com/mcp-use/mcp-use" alt="GitHub stars">
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

MCP-Use is the open-source SDK that empowers you to effortlessly integrate your Large Language Models (LLMs) with any MCP server, unlocking the power of tools and building custom agents without relying on closed-source solutions.

**Key Features:**

*   ✅ **Effortless Integration:** Connect any LLM (e.g., OpenAI, Anthropic, Groq) supporting tool calling to any MCP server.
*   ✅ **Versatile Tool Access:**  Enable your agents with a wide range of tools, including web browsing, file operations, and more.
*   ✅ **Code Builder:** Quickly get started with the interactive [code builder](https://mcp-use.com/builder).
*   ✅ **HTTP Support:** Directly connect to MCP servers running on specific HTTP ports.
*   ✅ **Dynamic Server Selection:**  Intelligently choose the most appropriate MCP server.
*   ✅ **Multi-Server Support:** Leverage multiple MCP servers simultaneously within a single agent.
*   ✅ **Tool Access Control:** Restrict potentially dangerous tools like file system or network access.
*   ✅ **Custom Agent Building:** Build your own agents with any framework using the LangChain adapter.
*   ✅ **Sandboxed Execution:** Securely run MCP servers in a sandboxed environment for enhanced security and portability.
*   ✅ **Direct Tool Calls:** Call MCP server tools directly without an LLM.
*   ✅ **Real-time Streaming Output:** Receive incremental results, tool actions, and intermediate steps.

**Get Started:**

1.  **Installation:**

    ```bash
    pip install mcp-use
    ```

    Or install from source:

    ```bash
    git clone https://github.com/pietrozullo/mcp-use.git
    cd mcp-use
    pip install -e .
    ```

2.  **Install LangChain Providers:**

    ```bash
    # For OpenAI
    pip install langchain-openai

    # For Anthropic
    pip install langchain-anthropic
    ```

    For other providers, see [LangChain chat models documentation](https://python.langchain.com/docs/integrations/chat/) and add your API keys to your `.env` file:

    ```
    OPENAI_API_KEY=
    ANTHROPIC_API_KEY=
    ```

3.  **Quick Start Example:**

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

**Further Exploration:**

*   [Website](https://mcp-use.com/)
*   [Documentation](https://docs.mcp-use.com/)
*   [TypeScript Version](https://github.com/mcp-use/mcp-use-ts)

**Example Use Cases:**

*   [Web Browsing with Playwright](#web-browsing-with-playwright)
*   [Airbnb Search](#airbnb-search)
*   [Blender 3D Creation](#blender-3d-creation)

**Learn More:**

*   [HTTP Connection Example](#http-connection-example)
*   [Multi-Server Support](#multi-server-support)
*   [Dynamic Server Selection (Server Manager)](#dynamic-server-selection-server-manager)
*   [Tool Access Control](#tool-access-control)
*   [Sandboxed Execution](#sandboxed-execution)
*   [Direct Tool Calls](#direct-tool-calls-without-llm)
*   [Build a Custom Agent:](#build-a-custom-agent)
*   [Debugging](#debugging)

**Contributing & Support**

*   We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).
*   Join the community on [Discord](https://discord.gg/XkNkSkMz3V).

**Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

**Requirements:**

*   Python 3.11+
*   MCP implementation (like Playwright MCP)
*   LangChain and appropriate model libraries (OpenAI, Anthropic, etc.)

**License:** MIT

**Citation:**

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

**[Back to Top](#)**