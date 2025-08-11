# Model Context Protocol Servers

Unlock the power of Large Language Models (LLMs) with secure and controlled access to tools and data using the Model Context Protocol (MCP). This repository provides reference implementations, community-built servers, and essential resources to help you build, deploy, and integrate MCP servers.  Explore the [original repository](https://github.com/modelcontextprotocol/servers) for more details.

## Key Features

*   **Reference Implementations:** Explore working examples of MCP servers to understand how to securely provide LLMs with access to tools and data.
*   **Extensible:** Showcase the versatility and adaptability of MCP across various use cases by implementing the MCP SDKs.
*   **Community-Driven:** Discover a rich ecosystem of community-built servers addressing diverse needs.
*   **Cross-Language SDKs:** Access easy to use SDKs to implement MCP servers using various programming languages.

## SDKs

Quickly build your own MCP servers with language-specific SDKs:

*   [C# MCP SDK](https://github.com/modelcontextprotocol/csharp-sdk)
*   [Go MCP SDK](https://github.com/modelcontextprotocol/go-sdk)
*   [Java MCP SDK](https://github.com/modelcontextprotocol/java-sdk)
*   [Kotlin MCP SDK](https://github.com/modelcontextprotocol/kotlin-sdk)
*   [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
*   [Ruby MCP SDK](https://github.com/modelcontextprotocol/ruby-sdk)
*   [Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk)
*   [Swift MCP SDK](https://github.com/modelcontextprotocol/swift-sdk)
*   [TypeScript MCP SDK](https://github.com/modelcontextprotocol/typescript-sdk)

## Reference Servers

Explore example servers demonstrating the core functionalities of MCP:

*   **[Everything](src/everything)** - Comprehensive reference server with prompts, resources, and tools.
*   **[Fetch](src/fetch)** - Web content fetching and conversion for streamlined LLM interactions.
*   **[Filesystem](src/filesystem)** - Secure file operations with customizable access control.
*   **[Git](src/git)** - Tools to read, search, and manipulate Git repositories.
*   **[Memory](src/memory)** - Persistent memory system based on knowledge graphs.
*   **[Sequential Thinking](src/sequentialthinking)** - Dynamic, reflective problem-solving via thought sequences.
*   **[Time](src/time)** - Time and timezone conversion capabilities.

### Archived Servers

These servers are archived and available at the [servers-archived](https://github.com/modelcontextprotocol/servers-archived) repository.

## Third-Party Servers

### Official Integrations

Leverage production-ready MCP servers from leading companies:

*   [Full list of official integrations](https://github.com/modelcontextprotocol/servers#official-integrations)

### Community Servers

Explore a growing collection of community-developed servers, demonstrating the versatility of MCP across a wide range of applications:

*   [Full list of community servers](https://github.com/modelcontextprotocol/servers#community-servers)

## 🚀 Getting Started

### Using MCP Servers in this Repository

TypeScript-based servers in this repository can be used directly with `npx`.

For example, this will start the [Memory](src/memory) server:

```sh
npx -y @modelcontextprotocol/server-memory
```

Python-based servers in this repository can be used directly with [`uvx`](https://docs.astral.sh/uv/concepts/tools/) or [`pip`](https://pypi.org/project/pip/). `uvx` is recommended for ease of use and setup.

For example, this will start the [Git](src/git) server:

```sh
# With uvx
uvx mcp-server-git

# With pip
pip install mcp-server-git
python -m mcp_server_git
```

Follow [these](https://docs.astral.sh/uv/getting-started/installation/) instructions to install `uv` / `uvx` and [these](https://pip.pypa.io/en/stable/installation/) to install `pip`.

### Using an MCP Client

To connect to your new server and make it useful, it should be configured into an MCP client. For example, here's the Claude Desktop configuration to use the above server:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

## 🛠️ Creating Your Own Server

Learn to build your own MCP server with the official documentation: [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction).

## 🤝 Contributing

Contribute to the MCP ecosystem by reviewing the [CONTRIBUTING.md](CONTRIBUTING.md) for information.

## 🔒 Security

Report security vulnerabilities via [SECURITY.md](SECURITY.md).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Community

*   [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions)