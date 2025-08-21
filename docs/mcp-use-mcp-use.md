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
  <a href="https://www.producthunt.com/products/mcp-use?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-mcp&#0045;use" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1002629&theme=neutral&t=1754609432704" alt="mcp&#0045;use - Open&#0032;source&#0032;SDK&#0032;and&#0032;infra&#0032;for&#0032;MCP&#0032;servers&#0032;&#0038;&#0032;agents | Product Hunt" style="width: 220px; height: 54px;" width="250" height="54" /></a>
  <p>👆 <em>Click to upvote and leave a comment!</em></p>
</div>
</div>

# MCP-Use: The Open-Source Framework for Building Custom Agents with LLMs and MCP Servers

**Easily connect any Large Language Model (LLM) to any MCP server with MCP-Use, empowering you to build powerful, customizable agents with tool access.** ([View on GitHub](https://github.com/mcp-use/mcp-use))

[![GitHub stars](https://img.shields.io/github/stars/pietrozullo/mcp-use?style=social)](https://github.com/pietrozullo/mcp-use/stargazers)
[![PyPI Downloads](https://static.pepy.tech/badge/mcp-use)](https://pypi.org/project/mcp_use/)
[![PyPI Version](https://img.shields.io/pypi/v/mcp_use.svg)](https://pypi.org/project/mcp_use/)
[![TypeScript](https://img.shields.io/badge/TypeScript-mcp--use-3178C6?logo=typescript&logoColor=white)](https://github.com/mcp-use/mcp-use-ts)
[![License](https://img.shields.io/github/license/pietrozullo/mcp-use)](https://github.com/pietrozullo/mcp-use/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mcp--use.com-blue)](https://docs.mcp-use.com)
[![Website](https://img.shields.io/badge/website-mcp--use.com-blue)](https://mcp-use.com)
[![Twitter Follow - Pietro](https://img.shields.io/twitter/follow/Pietro?style=social)](https://x.com/pietrozullo)
[![Twitter Follow - Luigi](https://img.shields.io/twitter/follow/Luigi?style=social)](https://x.com/pederzh)
[![Discord](https://dcbadge.limes.pink/api/server/XkNkSkMz3V?style=flat)](https://discord.gg/XkNkSkMz3V)


## Key Features

*   ✅ **LLM Agnostic:** Works seamlessly with any LangChain-supported LLM that supports tool calling (e.g., OpenAI, Anthropic, Groq, Llama).
*   🌐 **Open Source:** Built with open-source principles.
*   ⚙️ **Easy Setup:** Get started with just a few lines of code.
*   🔗 **HTTP Support:** Connect directly to MCP servers via HTTP, enabling integration with web-based tools.
*   🧩 **Multi-Server Support:** Leverage multiple MCP servers simultaneously within a single agent.
*   🛡️ **Tool Access Control:** Restrict access to potentially dangerous tools like file system or network access for enhanced security.
*   🤖 **Sandboxed Execution:** Run MCP servers in a sandboxed environment using E2B, eliminating the need for local dependencies and improving security.
*   🚀 **Direct Tool Calls:** Programmatically call MCP server tools directly, bypassing the need for an LLM when needed.
*   💻 **Custom Agents:** Build your own agents with any framework using the LangChain adapter.
*   💡 **Code Builder:** Explore MCP capabilities and generate starter code with the interactive [code builder](https://mcp-use.com/builder).
*   ⚡ **Streaming Output:** Get real-time results with asynchronous streaming of agent output, enabling dynamic UIs.
*   🧠 **Dynamic Server Selection:** Intelligent server selection based on tool requirements.

## Installation

Install using pip:

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

MCP-Use uses LangChain for LLM integration. You'll need to install the LangChain provider package for your chosen LLM.  For example:

```bash
# For OpenAI
pip install langchain-openai

# For Anthropic
pip install langchain-anthropic
```

Add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

> **Important**: Only models with tool calling capabilities can be used with mcp_use. Make sure your chosen model supports function calling or tool use.

## Quick Start

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

## Examples

### Web Browsing with Playwright

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

### Airbnb Search

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

### Blender 3D Creation

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

## Configuration

### HTTP Connection Example

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

### Dynamic Server Selection (Server Manager)

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

## Advanced Usage

### Streaming Agent Output

MCP-Use supports asynchronous streaming of agent output using the `astream` method on `MCPAgent`.

```python
async for chunk in agent.astream("Find the best restaurant in San Francisco"):
    print(chunk["messages"], end="", flush=True)
```

### Direct Tool Calls

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

### Build a Custom Agent:

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

### Enabling Debug Mode

```bash
# Level 1: Show INFO level messages
DEBUG=1 python3.11 examples/browser_use.py

# Level 2: Show DEBUG level messages (full verbose output)
DEBUG=2 python3.11 examples/browser_use.py
```

Or use the following environment variable to the desired logging level:
```bash
export MCP_USE_DEBUG=1 # or 2
```

### Agent-Specific Verbosity

```python
# Create agent with increased verbosity
agent = MCPAgent(
    llm=your_llm,
    client=your_client,
    verbose=True  # Only shows debug messages from the agent
)
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pietrozullo/mcp-use&type=Date)](https://www.star-history.com/#pietrozullo/mcp-use&Date)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](https://github.com/mcp-use/mcp-use/blob/main/LICENSE)

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