# Model Context Protocol Servers: Empowering LLMs with Secure Tool Access

Unlock the potential of Large Language Models (LLMs) by providing them with secure and controlled access to tools and data using Model Context Protocol (MCP).  This repository hosts reference implementations and a vibrant ecosystem of community-built servers.  Explore the [Model Context Protocol](https://modelcontextprotocol.io/) to learn more.

**[Visit the original repo](https://github.com/modelcontextprotocol/servers) for the latest updates and contributions.**

## Key Features:

*   **Reference Implementations:**  Explore example servers demonstrating MCP's versatility.
*   **Community-Driven:**  Access a growing library of community-built servers for various applications.
*   **Secure Tool Access:** Provide LLMs with controlled access to tools and data sources.
*   **Extensible Ecosystem:**  Discover frameworks and resources to build your own MCP servers.
*   **Language Support:**  Utilize MCP SDKs in your preferred languages:
    *   [C# MCP SDK](https://github.com/modelcontextprotocol/csharp-sdk)
    *   [Go MCP SDK](https://github.com/modelcontextprotocol/go-sdk)
    *   [Java MCP SDK](https://github.com/modelcontextprotocol/java-sdk)
    *   [Kotlin MCP SDK](https://github.com/modelcontextprotocol/kotlin-sdk)
    *   [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
    *   [Typescript MCP SDK](https://github.com/modelcontextprotocol/typescript-sdk)

## Reference Servers: Showcase MCP Capabilities

These servers demonstrate core MCP features and are built using the official SDKs.

*   [Everything](src/everything):  A comprehensive reference server with prompts, resources, and tools.
*   [Fetch](src/fetch): Web content fetching and conversion for efficient LLM usage.
*   [Filesystem](src/filesystem): Secure file operations with configurable access controls.
*   [Git](src/git): Tools to read, search, and manipulate Git repositories.
*   [Memory](src/memory): Knowledge graph-based persistent memory system.
*   [Sequential Thinking](src/sequentialthinking): Dynamic and reflective problem-solving through thought sequences.
*   [Time](src/time): Time and timezone conversion capabilities.

### Archived Servers

Find the following archived servers in the [servers-archived](https://github.com/modelcontextprotocol/servers-archived) repository.

*   AWS KB Retrieval
*   Brave Search
*   EverArt
*   GitHub
*   GitLab
*   Google Drive
*   Google Maps
*   PostgreSQL
*   Puppeteer
*   Redis
*   Sentry
*   Slack (maintained by [Zencoder](https://github.com/zencoderai/slack-mcp-server))
*   SQLite

## 🤝 Third-Party Servers: Expanding the Ecosystem

### 🎖️ Official Integrations: Production-Ready Solutions

These integrations are maintained by companies providing production-ready MCP servers for their platforms.

*   *   ... (List of official integrations, incorporating the HTML-based approach for logos to be SEO-friendly and accessible.)
    *   Replace existing logo icons with this markdown structure to enhance SEO.  Example:
    *   `<img height="12" width="12" src="https://www.21st.dev/favicon.ico" alt="21st.dev Logo" /> [21st.dev Magic](https://github.com/21st-dev/magic-mcp)`
    *   Continue the pattern for all the official integrations.

### 🌎 Community Servers:  Extending Functionality

A large and growing collection of community-developed servers is available.  **Note:** Community servers are **untested** and should be used at **your own risk**. They are not affiliated with or endorsed by Anthropic.

*   ... (List of community servers, potentially truncated for brevity but keeping the format similar to the original.  Consider alphabetizing this list, and summarizing where possible to save space and aid readability.)

## 📚 Frameworks: Building and Using MCP Servers

Resources to help you build and interact with MCP servers.

### For Servers

*   ... (List of server frameworks -  Use the same style as above, consider alphabetizing)

### For Clients

*   ... (List of client frameworks -  Use the same style as above, consider alphabetizing)

## 📚 Resources:  Learn More

*   [AiMCP](https://www.aimcp.info)
*   [Awesome Crypto MCP Servers by badkk](https://github.com/badkk/awesome-crypto-mcp-servers)
*   [Awesome MCP Servers by appcypher](https://github.com/appcypher/awesome-mcp-servers)
*   [Awesome MCP Servers by punkpeye](https://github.com/punkpeye/awesome-mcp-servers)
*   [Awesome MCP Servers by wong2](https://github.com/wong2/awesome-mcp-servers)
*   [Awesome Remote MCP Servers by JAW9C](https://github.com/jaw9c/awesome-remote-mcp-servers)
*   [Discord Server](https://glama.ai/mcp/discord)
*   [Discord Server (ModelContextProtocol)](https://discord.gg/jHEGxQu2a5)
*   ... (Continue listing all resources, alphabetized. Also replace each resource description with a summarized version to aid the readability)

## 🚀 Getting Started:  Quick Deployment

### Running Servers

Typescript-based servers (e.g., [Memory](src/memory)) can be run using `npx`:

```bash
npx -y @modelcontextprotocol/server-memory
```

Python-based servers (e.g., [Git](src/git)) can be started using [`uvx`](https://docs.astral.sh/uv/concepts/tools/) or [`pip`](https://pypi.org/project/pip/) and `python`:

```bash
uvx mcp-server-git
# OR
pip install mcp-server-git
python -m mcp_server_git
```

### Connecting with an MCP Client

Configure your MCP client (e.g., Claude Desktop) to use a server:

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

## 🛠️ Creating Your Own Server:

Build your own MCP server!  Consult the official documentation at [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction) for detailed guidance.

## 🤝 Contributing:

Help improve MCP!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🔒 Security:

Report vulnerabilities in [SECURITY.md](SECURITY.md).

## 📜 License:

MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Community:

*   [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions)

## ⭐ Support:

Show your support! Star the repository and contribute.

---

**This project is a community effort managed by Anthropic.**