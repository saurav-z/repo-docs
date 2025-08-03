<div align="center">
<div align="center" style="margin: 0 auto; max-width: 80%;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="static/logo_black.svg">
    <img alt="mcp use logo" src="./static/logo-white.svg" width="80%" style="margin: 20px auto;">
  </picture>
</div>

<br>

# Connect any LLM to any MCP server

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

## Unleash the Power of LLMs with MCP: Connect Any LLM to Any MCP Server with Ease

[mcp-use](https://github.com/mcp-use/mcp-use) is the open-source powerhouse for connecting your favorite Large Language Models (LLMs) to any Model Context Protocol (MCP) server, empowering you to build custom agents with access to powerful tools and services without relying on closed-source solutions.

**Key Features:**

*   ✅ **LLM Agnostic:** Compatible with any LangChain-supported LLM that supports tool calling, including OpenAI, Anthropic, Groq, and Llama models.
*   ✅ **Seamless Integration:** Effortlessly connect your LLMs to tools like web browsing, file operations, and more through MCP servers.
*   ✅ **Code Builder:** Get started quickly with the interactive [code builder](https://mcp-use.com/builder) to generate starter code.
*   ✅ **HTTP Support:** Direct connection to MCP servers running on specific HTTP ports.
*   ✅ **Dynamic Server Selection:**  Agents can intelligently select the most appropriate MCP server for a given task.
*   ✅ **Multi-Server Support:** Utilize multiple MCP servers simultaneously within a single agent.
*   ✅ **Tool Access Control:** Restrict access to potentially dangerous tools, such as file system or network access.
*   ✅ **Custom Agent Building:** Build your own custom agents using the LangChain adapter, or create new adapters for your framework.
*   ✅ **Sandboxed Execution:** Run MCP servers in a sandboxed environment for enhanced security and resource efficiency.
*   ✅ **Streaming Output:** Get real-time feedback with asynchronous streaming of agent output.

**Get Started Quickly**

*   **Website:**  [mcp-use.com](https://mcp-use.com/) - Build and deploy agents.
*   **Documentation:** [mcp-use docs](https://docs.mcp-use.com/) - Dive into the library's capabilities.
*   **TypeScript version:**  [mcp-use-ts](https://github.com/mcp-use/mcp-use-ts)

### Installing

With pip:

```bash
pip install mcp-use
```

Or install from source:

```bash
git clone https://github.com/pietrozullo/mcp-use.git
cd mcp-use
pip install -e .
```

**Important:**  Install the appropriate LangChain provider package for your chosen LLM (e.g., `pip install langchain-openai` for OpenAI). Ensure your chosen model supports tool calling.  Add your API keys to a `.env` file: `OPENAI_API_KEY=`

**Quickstart Code:**

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

### Streaming Agent Output

Receive incremental results with `agent.astream(query)`:

```python
async for chunk in agent.astream("Find the best restaurant in San Francisco"):
    print(chunk["messages"], end="", flush=True)
```

**Example: Streaming in Practice**

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    load_dotenv()
    client = MCPClient.from_config_file("browser_mcp.json")
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    async for chunk in agent.astream("Look for job at nvidia for machine learning engineer."):
        print(chunk["messages"], end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

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

### Configuration File Support

Easily manage MCP server setups:

```python
import asyncio
from mcp_use import create_session_from_config

async def main():
    # Create an MCP session from a config file
    session = create_session_from_config("mcp-config.json")

    # Initialize the session
    await session.initialize()

    # Use the session...

    # Disconnect when done
    await session.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### HTTP Connection Example

Connect to MCP servers via HTTP:

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

### Multi-Server Support

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

Manually select servers:

```python
result = await agent.run(
    "Search for Airbnb listings in Barcelona",
    server_name="airbnb" # Explicitly use the airbnb server
)

result_google = await agent.run(
    "Find restaurants near the first result using Google Search",
    server_name="playwright" # Explicitly use the playwright server
)
```

### Dynamic Server Selection (Server Manager)

Enable the Server Manager for intelligent server selection:

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

### Tool Access Control

Restrict agent tool access for security:

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

### Sandboxed Execution

Run MCP servers in a sandboxed environment using E2B's cloud infrastructure:

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

**Sandbox Options:**

| Option                 | Description                                                                              | Default               |
| ---------------------- | ---------------------------------------------------------------------------------------- | --------------------- |
| `api_key`              | E2B API key. Required - can be provided directly or via E2B_API_KEY environment variable | None                  |
| `sandbox_template_id`  | Template ID for the sandbox environment                                                  | "base"                |
| `supergateway_command` | Command to run supergateway                                                              | "npx -y supergateway" |

### Build a Custom Agent

Create your own agent using the LangChain adapter:

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

### Debugging

Increase log verbosity:

```bash
DEBUG=1 python3.11 examples/browser_use.py  # INFO level
DEBUG=2 python3.11 examples/browser_use.py  # DEBUG level
```
or
```bash
export MCP_USE_DEBUG=1 # or 2
```

or
```python
agent = MCPAgent(..., verbose=True)  # Agent-specific verbosity
```
### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

### Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md).

### Contributors

<a href="https://github.com/mcp-use/mcp-use/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mcp-use/mcp-use" />
</a>

### Top Starred Dependents

<!-- gh-dependents-info-used-by-start -->

<table>
  <tr>
    <th width="400">Repository</th>
    <th>Stars</th>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/170207473?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/tavily-ai/meeting-prep-agent"><strong>tavily-ai/meeting-prep-agent</strong></a></td>
    <td>⭐ 127</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/205593730?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/Qingyon-AI/Revornix"><strong>Qingyon-AI/Revornix</strong></a></td>
    <td>⭐ 108</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/20041231?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/krishnaik06/MCP-CRASH-Course"><strong>krishnaik06/MCP-CRASH-Course</strong></a></td>
    <td>⭐ 57</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/892404?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/truemagic-coder/solana-agent-app"><strong>truemagic-coder/solana-agent-app</strong></a></td>
    <td>⭐ 30</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/8344498?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/schogini/techietalksai"><strong>schogini/techietalksai</strong></a></td>
    <td>⭐ 23</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/201161342?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/autometa-dev/whatsapp-mcp-voice-agent"><strong>autometa-dev/whatsapp-mcp-voice-agent</strong></a></td>
    <td>⭐ 22</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/100749943?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/Deniscartin/mcp-cli"><strong>Deniscartin/mcp-cli</strong></a></td>
    <td>⭐ 18</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/6688805?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/innovaccer/Healthcare-MCP"><strong>innovaccer/Healthcare-MCP</strong></a></td>
    <td>⭐ 12</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/6764390?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/elastic/genai-workshops"><strong>elastic/genai-workshops</strong></a></td>
    <td>⭐ 10</td>
  </tr>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/u/68845761?s=40&v=4" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> <a href="https://github.com/entbappy/MCP-Tutorials"><strong>entbappy/MCP-Tutorials</strong></a></td>
    <td>⭐ 6</td>
  </tr>
</table>

<!-- gh-dependents-info-used-by-end -->

### Requirements

*   Python 3.11+
*   MCP implementation (e.g., Playwright MCP)
*   LangChain and appropriate model libraries (e.g., OpenAI, Anthropic)

### License

MIT

### Citation

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