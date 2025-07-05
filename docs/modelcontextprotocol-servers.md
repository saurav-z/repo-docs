# Model Context Protocol Servers: Unleash the Power of AI with Secure Tool Access

**Empower your Large Language Models (LLMs) with secure and controlled access to tools and data sources using the Model Context Protocol (MCP).** This repository provides reference implementations and a vibrant community of servers, enabling you to extend the capabilities of your AI applications. Explore the **[original repository](https://github.com/modelcontextprotocol/servers)** for more details.

## Key Features

*   **Reference Implementations:** Explore working examples of MCP servers, showcasing the protocol's versatility.
*   **SDK Support:** Utilize official SDKs for [C#](https://github.com/modelcontextprotocol/csharp-sdk), [Java](https://github.com/modelcontextprotocol/java-sdk), [Kotlin](https://github.com/modelcontextprotocol/kotlin-sdk), [Python](https://github.com/modelcontextprotocol/python-sdk), and [Typescript](https://github.com/modelcontextprotocol/typescript-sdk).
*   **Extensible & Versatile:** Build MCP servers for a wide range of use cases, from web content fetching to secure file operations and complex integrations.
*   **Community-Driven Ecosystem:** Benefit from a growing collection of third-party servers, including official integrations and community contributions.

## 🚀 Getting Started

Get up and running with MCP servers quickly.

### Using MCP Servers in this Repository

**For Typescript-based servers, use `npx`:**

```bash
npx -y @modelcontextprotocol/server-memory
```

**For Python-based servers, use `uvx` or `pip`:**

```bash
# Using uvx
uvx mcp-server-git

# Using pip
pip install mcp-server-git
python -m mcp_server_git
```

### Configuring an MCP Client

Integrate MCP servers into your preferred client (like Claude Desktop) using configuration files.

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

These servers demonstrate the core functionality of MCP and the use of the official SDKs:

*   **Everything:** A versatile test server with prompts, resources, and tools.
*   **Fetch:** Retrieves and converts web content for efficient LLM use.
*   **Filesystem:** Provides secure file operations with access control.
*   **Git:** Offers tools for interacting with Git repositories.
*   **Memory:** Utilizes a knowledge graph for persistent memory.
*   **Sequential Thinking:** Facilitates dynamic problem-solving through thought sequences.
*   **Time:** Converts time and timezone information.

### Archived

Archived servers can be found at [servers-archived](https://github.com/modelcontextprotocol/servers-archived).

## 🤝 Third-Party Servers

Explore a dynamic ecosystem of third-party servers that expand the capabilities of MCP, including:

### 🎖️ Official Integrations

Official integrations maintained by platform providers:

*   *   **[21st.dev Magic](https://github.com/21st-dev/magic-mcp):** Create crafted UI components inspired by the best 21st.dev design engineers.
    *   **[ActionKit by Paragon](https://github.com/useparagon/paragon-mcp):** Connect to 130+ SaaS integrations (e.g. Slack, Salesforce, Gmail) with Paragon’s [ActionKit](https://www.useparagon.com/actionkit) API.
*   *   And many more integrations listed in the original README, see link above.

### 🌎 Community Servers

A wide range of community-developed servers, for example:

*   *   **[1Panel](https://github.com/1Panel-dev/mcp-1panel):** MCP server implementation that provides 1Panel interaction.
    *   **[A2A](https://github.com/GongRzhe/A2A-MCP-Server):** An MCP server that bridges the Model Context Protocol (MCP) with the Agent-to-Agent (A2A) protocol, enabling MCP-compatible AI assistants (like Claude) to seamlessly interact with A2A agents.
*   *   And many more community integrations listed in the original README, see link above.

## 🛠️ Build Your Own Server

Learn how to create custom MCP servers using the comprehensive documentation at [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction).

## 🤝 Contribute

Help improve this project by contributing to [CONTRIBUTING.md](CONTRIBUTING.md).

## 🔒 Security

Report security vulnerabilities via [SECURITY.md](SECURITY.md).

## 📜 License

This project is licensed under the MIT License ([LICENSE](LICENSE)).

## 💬 Community

Engage with the MCP community in [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions).

## ⭐ Support

Show your support by starring the repository and contributing new servers or enhancements!

---

*Managed by Anthropic, but built with community contributions.*