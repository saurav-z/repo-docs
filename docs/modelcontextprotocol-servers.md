# Model Context Protocol (MCP) Servers: Empowering AI with Secure Tool and Data Access

**Unlock the power of Large Language Models (LLMs) by giving them secure and controlled access to tools and data sources with the Model Context Protocol (MCP).** This repository provides reference implementations and a vibrant ecosystem of community-built servers to enhance your AI applications. [Explore the original repository](https://github.com/modelcontextprotocol/servers) for details and the latest updates.

**Key Features:**

*   **Secure Access:** Grant LLMs controlled access to external resources and data.
*   **Reference Implementations:** Explore example servers demonstrating MCP capabilities.
*   **SDKs & Frameworks:** Leverage existing SDKs and frameworks to build your own servers.
*   **Community-Driven:** Benefit from a growing ecosystem of community-developed and maintained servers.
*   **Extensible:** Extend MCP functionality with new servers tailored to your specific needs.

## 💡 Getting Started

### 🚀 Deploying a Reference Server

Easily run TypeScript-based servers with `npx`:

```bash
npx -y @modelcontextprotocol/server-memory
```

For Python-based servers, use `uvx` (recommended) or `pip`:

```bash
# Using uvx
uvx mcp-server-git

# Using pip
pip install mcp-server-git
python -m mcp_server_git
```

### 💻 Configuring an MCP Client

Configure your MCP client (e.g., Claude Desktop) by specifying the server command and arguments:

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

## 🌟 Reference Servers

These servers showcase MCP features using the official SDKs:

*   **Everything:** Test server with prompts, resources, and tools.
*   **Fetch:** Web content retrieval and conversion.
*   **Filesystem:** Secure file operations.
*   **Git:** Git repository interaction.
*   **Memory:** Persistent knowledge graph memory system.
*   **Sequential Thinking:** Dynamic problem-solving.
*   **Time:** Time and timezone conversion.

## 🤝 Third-Party Servers

A comprehensive list of community-built servers, categorized for easy navigation:

### 🎖️ Official Integrations

Official integrations maintained by companies building production ready MCP servers for their platforms.
(Listed in the original README)

### 🌎 Community Servers

A growing collection of community-developed servers (listed in the original README). **Use these at your own risk.**

## 🛠️ Build Your Own Server

For in-depth guidance on building your own MCP server, explore the official documentation: [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction).

## 📚 Resources

*   Frameworks for easier server and client development
*   Directories to discover more MCP servers
*   Related libraries, articles, and news
*   A large list of community MCP servers with links to documentation and code.
(Listed in the original README)

## 🤝 Contributing

Contribute to the MCP ecosystem! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔒 Security

Report security vulnerabilities in [SECURITY.md](SECURITY.md).

## 📜 License

This project is MIT licensed.

## 💬 Community

*   [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions)