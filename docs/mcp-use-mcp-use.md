<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-gh.jpg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo-gh.jpg">
    <img alt="mcp use logo" src="./static/logo-gh.jpg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<h1 align="center">🚀 MCP-Use: Unleash the Power of LLMs with Custom MCP Agents</h1>

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

<br>

**MCP-Use empowers developers to connect any Large Language Model (LLM) to any Model Context Protocol (MCP) server, enabling the creation of powerful, customizable agents with tool access.**

<br>

## Key Features

*   **🛠️ Connect Any LLM:** Seamlessly integrate with any LLM supported by LangChain (OpenAI, Anthropic, Groq, Llama, etc.).
*   **🔗 Open Source & Flexible:** Build custom MCP agents without relying on closed-source or application clients.
*   **🌐 Code Builder:** Quickly explore MCP capabilities and generate starter code with the interactive [code builder](https://mcp-use.com/builder).
*   **💻 HTTP Support:** Directly connect to MCP servers running on specific HTTP ports.
*   **🔄 Ease of Use:** Create your first MCP-capable agent with as little as six lines of code.
*   **📦 Multi-Server Support:** Utilize multiple MCP servers simultaneously within a single agent.
*   **🔒 Tool Access Control:** Restrict tool access for enhanced security and control.
*   **💡 Dynamic Server Selection:**  Intelligently choose the most appropriate MCP server for each task.
*   **🚀  Asynchronous Streaming:** Get real-time feedback and progress with asynchronous streaming for agent output using the `astream` method on `MCPAgent`.
*   **🧩 Build Custom Agents:** Leverage a LangChain adapter to create custom agents and integrate with your preferred frameworks.
*   **📦 Sandboxed Execution:** Run MCP servers in a sandboxed environment using E2B's cloud infrastructure for secure and resource-efficient operation.
*   **🔨 Direct Tool Calls:** Call MCP server tools directly without an LLM for programmatic control.

## Getting Started

### Installation

```bash
pip install mcp-use
```

Or install from source:

```bash
git clone https://github.com/mcp-use/mcp-use.git
cd mcp-use
pip install -e .
```

### Installing LangChain Providers

mcp\_use works with various LLM providers through LangChain. You'll need to install the appropriate LangChain provider package for your chosen LLM. For example:

```bash
# For OpenAI
pip install langchain-openai

# For Anthropic
pip install langchain-anthropic
```

For other providers, check the [LangChain chat models documentation](https://python.langchain.com/docs/integrations/chat/) and add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

> **Important**: Only models with tool calling capabilities can be used with mcp_use. Make sure your chosen model supports function calling or tool use.

### Quickstart Example

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

You can also add the servers configuration from a config file like this:

```python
client = MCPClient.from_config_file(
        os.path.join("browser_mcp.json")
    )
```

## Example Use Cases

Explore the power of MCP-Use with these practical examples:

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

## Configuration Support

*   **HTTP Connection:** Easily connect to MCP servers over HTTP.

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

*   **Multi-Server Support:** Leverage multiple MCP servers simultaneously.

    *   **Configuration Example:**

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

    *   **Usage:**

        ```python
        # Example: Manually selecting a server for a specific task
        result = await agent.run(
            "Search for Airbnb listings in Barcelona",
            server_name="airbnb" # Explicitly use the airbnb server
        )

        result_google = await agent.run(
            "Find restaurants near the first result using Google Search",
            server_name="playwright" # Explicitly use the playwright server
        )
        ```
    *   **Dynamic Server Selection:**  Use the Server Manager for intelligent server selection.

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

*   **Tool Access Control:** Restrict tool access for security.

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

*   **Sandboxed Execution:**  Run MCP servers in a secure, isolated environment with E2B.

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

*   **Direct Tool Calls (Without LLM):**

    ```python
    import asyncio
    from mcp_use import MCPClient

    async def call_tool_example():
        config = {
            "mcpServers": {
                "everything": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-everything"],
                }
            }
        }

        client = MCPClient.from_dict(config)

        try:
            await client.create_all_sessions()
            session = client.get_session("everything")

            # Call tool directly
            result = await session.call_tool(
                name="add",
                arguments={"a": 1, "b": 2}
            )

            print(f"Result: {result.content[0].text}")  # Output: 3

        finally:
            await client.close_all_sessions()

    if __name__ == "__main__":
        asyncio.run(call_tool_example())
    ```

*   **Build a Custom Agent:**

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

## Debugging

MCP-Use offers robust debugging capabilities:

*   **Environment Variable:** Use `DEBUG=1` (INFO) or `DEBUG=2` (DEBUG) to control logging verbosity.
*   **Programmatic Debugging:** Use `mcp_use.set_debug()` in your code.
*   **Agent Verbosity:**  Set `verbose=True` in `MCPAgent` for agent-specific debug output.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

## Contributing

We welcome contributions!  Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## [GitHub Repository](https://github.com/mcp-use/mcp-use)

<br>

<!-- gh-dependents-info-used-by-start -->

## Top Starred Dependents

<table>
  <tr>
    <th width="400">Repository</th>
    <th>Stars</th>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/38653995?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/patchy631/ai-engineering-hub"><strong>patchy631/ai-engineering-hub</strong></a></td>
    <td>⭐ 17384</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/170207473?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/tavily-ai/meeting-prep-agent"><strong>tavily-ai/meeting-prep-agent</strong></a></td>
    <td>⭐ 131</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/164294848?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/buildfastwithai/gen-ai-experiments"><strong>buildfastwithai/gen-ai-experiments</strong></a></td>
    <td>⭐ 100</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/187057607?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/hud-evals/hud-python"><strong>hud-evals/hud-python</strong></a></td>
    <td>⭐ 78</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/20041231?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/krishnaik06/MCP-CRASH-Course"><strong>krishnaik06/MCP-CRASH-Course</strong></a></td>
    <td>⭐ 64</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/54944174?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/larksuite/lark-samples"><strong>larksuite/lark-samples</strong></a></td>
    <td>⭐ 35</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/892404?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/truemagic-coder/solana-agent-app"><strong>truemagic-coder/solana-agent-app</strong></a></td>
    <td>⭐ 30</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/8344498?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/schogini/techietalksai"><strong>schogini/techietalksai</strong></a></td>
    <td>⭐ 24</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/201161342?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/autometa-dev/whatsapp-mcp-voice-agent"><strong>autometa-dev/whatsapp-mcp-voice-agent</strong></a></td>
    <td>⭐ 23</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/100749943?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/Deniscartin/mcp-cli"><strong>Deniscartin/mcp-cli</strong></a></td>
    <td>⭐ 19</td>
  </tr>
</table>

<!-- gh-dependents-info-used-by-end -->

## Requirements

*   Python 3.11+
*   MCP implementation (e.g., Playwright MCP)
*   LangChain and relevant LLM libraries (e.g., OpenAI, Anthropic).

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