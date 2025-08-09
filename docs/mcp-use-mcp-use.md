<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<br>

# mcp-use: Build Powerful MCP Agents with Open Source

<p align="center">
<a href="https://www.producthunt.com/products/mcp-use?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-mcp&#0045;use" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1002629&theme=neutral&t=1754609432704" alt="mcp&#0045;use - Open&#0032;source&#0032;SDK&#0032;and&#0032;infra&#0032;for&#0032;MCP&#0032;servers&#0032;&#0038;&#0032;agents | Product Hunt" style="width: 150px; height: 32px;" width="150" height="32" /></a>
</p>
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
</div>

🌐 **mcp-use empowers you to connect any Large Language Model (LLM) to tools and build custom agents, opening up a world of possibilities beyond closed-source applications.**

Explore the [mcp-use repository](https://github.com/mcp-use/mcp-use) for the full source code and further details.

**Key Features:**

*   ✅ **Ease of Use:**  Quickly create MCP-capable agents with just a few lines of code.
*   🤖 **LLM Flexibility:** Compatible with any LangChain-supported LLM that supports tool use (OpenAI, Anthropic, Groq, Llama, etc.).
*   🌐 **Code Builder:** Interactive code builder at [mcp-use.com/builder](https://mcp-use.com/builder) to generate starter code.
*   🔗 **HTTP Support:** Direct connection to MCP servers via HTTP.
*   ⚙️ **Dynamic Server Selection:** Agents intelligently choose the best MCP server for a given task with Server Manager.
*   🧩 **Multi-Server Support:**  Use multiple MCP servers simultaneously.
*   🛡️ **Tool Restrictions:** Control agent access to tools for enhanced security (e.g., file system, network).
*   🔧 **Custom Agents:** Build your own agents using the LangChain adapter.
*   🚀 **Sandboxed Execution:** Run MCP servers in an isolated environment using E2B's cloud infrastructure.
*   ⚡ **Streaming Output:** Real-time results and progress reporting.

**Quick Start**

Install mcp-use using pip:

```bash
pip install mcp-use
```

or from source:

```bash
git clone https://github.com/pietrozullo/mcp-use.git
cd mcp-use
pip install -e .
```

**Install LangChain Providers**

Install the correct LangChain package for your desired LLM provider (e.g., `pip install langchain-openai`, `pip install langchain-anthropic`).  Remember to set your API keys in a `.env` file.

**Example: Basic Agent**

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

**Explore the documentation for more detailed usage and advanced configurations.**

**Example Use Cases**

*   **Web Browsing with Playwright:**

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

*   **Airbnb Search:**

    ```python
    import asyncio
    import os
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    from mcp_use import MCPAgent, MCPClient

    async def run_airbnb_example():
        # Load environment variables
        load_dotenv()

        # Create MCPClient with Airbnb configuration
        client = MCPClient.from_config_file(
            os.path.join(os.path.dirname(__file__), "airbnb_mcp.json")
        )

        # Create LLM - you can choose between different models
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

        # Create agent with the client
        agent = MCPAgent(llm=llm, client=client, max_steps=30)

        try:
            # Run a query to search for accommodations
            result = await agent.run(
                "Find me a nice place to stay in Barcelona for 2 adults "
                "for a week in August. I prefer places with a pool and "
                "good reviews. Show me the top 3 options.",
                max_steps=30,
            )
            print(f"\nResult: {result}")
        finally:
            # Ensure we clean up resources properly
            if client.sessions:
                await client.close_all_sessions()

    if __name__ == "__main__":
        asyncio.run(run_airbnb_example())
    ```

*   **Blender 3D Creation:**

    ```python
    import asyncio
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    from mcp_use import MCPAgent, MCPClient

    async def run_blender_example():
        # Load environment variables
        load_dotenv()

        # Create MCPClient with Blender MCP configuration
        config = {"mcpServers": {"blender": {"command": "uvx", "args": ["blender-mcp"]}}}
        client = MCPClient.from_dict(config)

        # Create LLM
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

        # Create agent with the client
        agent = MCPAgent(llm=llm, client=client, max_steps=30)

        try:
            # Run the query
            result = await agent.run(
                "Create an inflatable cube with soft material and a plane as ground.",
                max_steps=30,
            )
            print(f"\nResult: {result}")
        finally:
            # Ensure we clean up resources properly
            if client.sessions:
                await client.close_all_sessions()

    if __name__ == "__main__":
        asyncio.run(run_blender_example())
    ```

**Configuration Support**

*   **HTTP Connection:**  Connect to MCP servers over HTTP.
    ```python
    import asyncio
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from mcp_use import MCPAgent, MCPClient

    async def main():
        """Run the example using a configuration file."""
        # Load environment variables
        load_dotenv()

        config = {
            "mcpServers": {
                "http": {
                    "url": "http://localhost:8931/sse"
                }
            }
        }

        # Create MCPClient from config file
        client = MCPClient.from_dict(config)

        # Create LLM
        llm = ChatOpenAI(model="gpt-4o")

        # Create agent with the client
        agent = MCPAgent(llm=llm, client=client, max_steps=30)

        # Run the query
        result = await agent.run(
            "Find the best restaurant in San Francisco USING GOOGLE SEARCH",
            max_steps=30,
        )
        print(f"\nResult: {result}")

    if __name__ == "__main__":
        # Run the appropriate example
        asyncio.run(main())
    ```

*   **Multi-Server Support:**  Configure and connect to multiple MCP servers.
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

*   **Dynamic Server Selection (Server Manager):** Enables intelligent server selection.
    ```python
    import asyncio
    from mcp_use import MCPClient, MCPAgent
    from langchain_anthropic import ChatAnthropic

    async def main():
        # Create client with multiple servers
        client = MCPClient.from_config_file("multi_server_config.json")

        # Create agent with the client
        agent = MCPAgent(
            llm=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            client=client,
            use_server_manager=True  # Enable the Server Manager
        )

        try:
            # Run a query that uses tools from multiple servers
            result = await agent.run(
                "Search for a nice place to stay in Barcelona on Airbnb, "
                "then use Google to find nearby restaurants and attractions."
            )
            print(result)
        finally:
            # Clean up all sessions
            await client.close_all_sessions()

    if __name__ == "__main__":
        asyncio.run(main())
    ```

*   **Tool Access Control:** Restrict access to potentially dangerous tools.
    ```python
    import asyncio
    from mcp_use import MCPAgent, MCPClient
    from langchain_openai import ChatOpenAI

    async def main():
        # Create client
        client = MCPClient.from_config_file("config.json")

        # Create agent with restricted tools
        agent = MCPAgent(
            llm=ChatOpenAI(model="gpt-4"),
            client=client,
            disallowed_tools=["file_system", "network"]  # Restrict potentially dangerous tools
        )

        # Run a query with restricted tool access
        result = await agent.run(
            "Find the best restaurant in San Francisco"
        )
        print(result)

        # Clean up
        await client.close_all_sessions()

    if __name__ == "__main__":
        asyncio.run(main())
    ```

*   **Sandboxed Execution:** Run MCP servers in a secure, isolated environment.

    ```python
    import asyncio
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from mcp_use import MCPAgent, MCPClient
    from mcp_use.types.sandbox import SandboxOptions

    async def main():
        # Load environment variables (needs E2B_API_KEY)
        load_dotenv()

        # Define MCP server configuration
        server_config = {
            "mcpServers": {
                "everything": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-everything"],
                }
            }
        }

        # Define sandbox options
        sandbox_options: SandboxOptions = {
            "api_key": os.getenv("E2B_API_KEY"),  # API key can also be provided directly
            "sandbox_template_id": "base",  # Use base template
        }

        # Create client with sandboxed mode enabled
        client = MCPClient(
            config=server_config,
            sandbox=True,
            sandbox_options=sandbox_options,

        )

        # Create agent with the sandboxed client
        llm = ChatOpenAI(model="gpt-4o")
        agent = MCPAgent(llm=llm, client=client)

        # Run your agent
        result = await agent.run("Use the command line tools to help me add 1+1")
        print(result)

        # Clean up
        await client.close_all_sessions()

    if __name__ == "__main__":
        asyncio.run(main())
    ```

*   **Build a Custom Agent:** Use the LangChain adapter to create custom agents.

    ```python
    import asyncio
    from langchain_openai import ChatOpenAI
    from mcp_use.client import MCPClient
    from mcp_use.adapters.langchain_adapter import LangChainAdapter
    from dotenv import load_dotenv

    load_dotenv()


    async def main():
        # Initialize MCP client
        client = MCPClient.from_config_file("examples/browser_mcp.json")
        llm = ChatOpenAI(model="gpt-4o")

        # Create adapter instance
        adapter = LangChainAdapter()
        # Get LangChain tools with a single line
        tools = await adapter.create_tools(client)

        # Create a custom LangChain agent
        llm_with_tools = llm.bind_tools(tools)
        result = await llm_with_tools.ainvoke("What tools do you have available ? ")
        print(result)


    if __name__ == "__main__":
        asyncio.run(main())
    ```

**Debugging**

*   Use the `DEBUG` environment variable (e.g., `DEBUG=1 python your_script.py`) or the `verbose` parameter in `MCPAgent` initialization for detailed logging.

**Requirements**

*   Python 3.11+
*   MCP implementation (e.g., Playwright MCP)
*   LangChain and appropriate model libraries

**License**

MIT

**Citation**

```bibtex
@software{mcp_use2025,
  author = {Zullo, Pietro},
  title = {MCP-Use: MCP Library for Python},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pietrozullo/mcp-use}
}
```

**Contributing**

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Contributors**

[Contributor images from contributors.rocks]

**Dependents**

<!-- gh-dependents-info-used-by-start -->
[Dependents table, using github-dependents]
<!-- gh-dependents-info-used-by-end -->
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=732589b6-6850-4b8c-aa25-906c0979e426&page=README.md" />