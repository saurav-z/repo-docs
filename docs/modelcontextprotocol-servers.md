# Model Context Protocol Servers: Empowering AI with Controlled Tool Access

Unlock the potential of Large Language Models (LLMs) by giving them secure and controlled access to tools and data using Model Context Protocol (MCP). This repository, ([original repo](https://github.com/modelcontextprotocol/servers)), offers a collection of reference implementations, community-built servers, and helpful resources to help you build powerful AI-driven applications.

**Key Features:**

*   **Reference Implementations:** Explore example servers showcasing MCP's versatility.
*   **Community-Built Servers:** Discover a growing ecosystem of servers across various domains, expanding the capabilities of AI assistants.
*   **SDK Support:** Leverage official MCP SDKs for various languages including:
    *   C#
    *   Go
    *   Java
    *   Kotlin
    *   Python
    *   Typescript
*   **Secure Tool Access:** Grant LLMs controlled access to tools and data sources, ensuring safety and responsible AI usage.

## 🌟 Reference Servers

These servers demonstrate core MCP features and showcase the official SDKs.

*   **[Everything](src/everything)** - Comprehensive reference and test server.
*   **[Fetch](src/fetch)** - Retrieves web content for LLMs.
*   **[Filesystem](src/filesystem)** - Secure file operations with access control.
*   **[Git](src/git)** - Access and manipulate Git repositories.
*   **[Memory](src/memory)** - Knowledge graph-based persistent memory.
*   **[Sequential Thinking](src/sequentialthinking)** - Dynamic, reflective problem-solving.
*   **[Time](src/time)** - Time and timezone conversion capabilities.

### Archived Servers

These servers are archived and available at [servers-archived](https://github.com/modelcontextprotocol/servers-archived).

*   (List of archived servers - see original for full list)

## 🤝 Third-Party Servers

### 🎖️ Official Integrations

These integrations are maintained by companies building production-ready MCP servers for their platforms.

*   (List of official integrations with their respective logos/descriptions - see original for full list)

### 🌎 Community Servers

Explore a wide range of community-contributed servers, expanding MCP's capabilities.

>   **Note:** Community servers are provided "as is" and are not endorsed or maintained by Anthropic. Use with caution.
*   (List of community servers - see original for full list)

## 📚 Frameworks

### For Servers

*   ModelFetch (TypeScript)
*   EasyMCP (TypeScript)
*   FastAPI to MCP auto generator
*   FastMCP (TypeScript)
*   Foobara MCP Connector
*   Foxy Contexts (Go)
*   Higress MCP Server Hosting
*   MCP Declarative Java SDK
*   MCP-Framework (Typescript)
*   MCP Plexus
*   mcp\_sse (Elixir)
*   Next.js MCP Server Template (Typescript)
*   Quarkus MCP Server SDK (Java)
*   SAP ABAP MCP Server SDK
*   Spring AI MCP Server
*   Template MCP Server
*   AgentR Universal MCP SDK
*   Vercel MCP Adapter (Typescript)
*   Hermes MCP (Elixir)

### For Clients

*   codemirror-mcp
*   llm-analysis-assistant
*   MCP-Agent
*   Spring AI MCP Client
*   MCP CLI Client
*   OpenMCP Client

## 📚 Resources

Find additional resources for MCP:

*   AiMCP
*   Awesome Crypto MCP Servers by badkk
*   Awesome MCP Servers by appcypher
*   Awesome MCP Servers by punkpeye
*   Awesome MCP Servers by wong2
*   Awesome Remote MCP Servers by JAW9C
*   Discord Server by punkpeye
*   Discord Server (ModelContextProtocol) by Alex Andru
*   Klavis AI
*   MCP Badges
*   MCPRepository.com
*   mcp-cli
*   mcp-dockmaster
*   mcp-get
*   mcp-guardian
*   MCP Linker
*   mcp-manager
*   MCP Marketplace Web Plugin
*   mcp.natoma.id
*   mcp.run
*   MCPHub
*   MCP Servers Hub
*   MCPServers.com
*   MCP Servers Rating and User Reviews
*   MCP X Community
*   MCPHub by Jeamee
*   mcpm by Pathintegral
*   MCPVerse
*   MCP Servers Search
*   MCPWatch
*   mkinf
*   Open-Sourced MCP Servers Directory
*   OpenTools
*   PulseMCP
*   r/mcp
*   r/modelcontextprotocol
*   MCP.ing
*   MCP Hunt
*   Smithery
*   Toolbase
*   ToolHive
*   NetMind

## 🚀 Getting Started

### Using MCP Servers

*   **Typescript Servers:** Use `npx` to start servers. Example: `npx -y @modelcontextprotocol/server-memory`
*   **Python Servers:** Use `uvx` (recommended) or `pip`. Example (with `uvx`): `uvx mcp-server-git` or (with `pip`): `pip install mcp-server-git && python -m mcp_server_git`.

### Using an MCP Client

Configure your client (e.g., Claude Desktop) by adding server details to its configuration file. Examples provided in the original README.

## 🛠️ Creating Your Own Server

For information on building your own MCP server, consult the official documentation at [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction).

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 🔒 Security

See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities.

## 📜 License

This project is licensed under the MIT License (see [LICENSE](LICENSE) for details).

## 💬 Community

*   [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions)

## ⭐ Support

Show your support by starring the repository and contributing!

---

This project is managed by Anthropic but built with community contributions. The Model Context Protocol is open source, and we encourage your contributions!