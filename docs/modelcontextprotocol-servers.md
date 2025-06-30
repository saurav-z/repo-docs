# Model Context Protocol (MCP) Servers: Unlock Powerful LLM Integrations

**Connect Large Language Models (LLMs) to tools and data with secure, controlled access using the Model Context Protocol (MCP).** Explore a comprehensive collection of reference and community-built servers to enhance your AI applications. View the original repo here: [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)

**Key Features:**

*   **Reference Implementations:** Explore the versatility of MCP with example servers built using official SDKs.
*   **Community-Driven:** Discover a wide range of community-developed servers, extending MCP capabilities across various domains.
*   **Secure Access:** Grant your LLMs controlled and secure access to tools and data sources.
*   **Extensible:** Easily adapt and extend MCP to fit your specific needs.
*   **Open Source:** Built in partnership with Anthropic and open to community contributions.

## 🚀 Getting Started

### Running Reference Servers
Use `npx`, `uvx`, or `pip` to quickly launch reference servers.

**Example: Running the Memory Server (TypeScript)**
```bash
npx -y @modelcontextprotocol/server-memory
```

**Example: Running the Git Server (Python)**
```bash
# With uvx (Recommended)
uvx mcp-server-git

# With pip
pip install mcp-server-git
python -m mcp_server_git
```

### Configuring an MCP Client
Integrate servers with your MCP client (e.g., Claude Desktop) using configuration files.

**Example: Claude Desktop Configuration**
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

## 🌟 Core Reference Servers

These servers demonstrate MCP features and are implemented with the official MCP SDKs.

*   **Everything:** Reference / test server with prompts, resources, and tools
*   **Fetch:** Web content fetching and conversion for efficient LLM usage
*   **Filesystem:** Secure file operations with configurable access controls
*   **Git:** Tools to read, search, and manipulate Git repositories
*   **Memory:** Knowledge graph-based persistent memory system
*   **Sequential Thinking:** Dynamic and reflective problem-solving through thought sequences
*   **Time:** Time and timezone conversion capabilities

## 🤝 Third-Party Servers: Extending the Ecosystem

Explore a vast array of community and official integrations, expanding MCP's reach.

### 🎖️ Official Integrations

Official integrations, maintained by leading platforms, connect directly to your favorite tools.
*(A full list of official integrations including logos is in the original README.)*

### 🌎 Community Servers

Discover a thriving ecosystem of community-built servers, adding capabilities across various domains.
*(A full list of community integrations is in the original README.)*

## 🛠️ Build Your Own Server

Learn how to create your own MCP server by visiting the official documentation at [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction).

## 📚 Resources

*   **[AiMCP](https://www.aimcp.info):** Directory of MCP clients and servers
*   **[Awesome Crypto MCP Servers](https://github.com/badkk/awesome-crypto-mcp-servers):** Curated list of MCP servers
*   **[Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers):** Curated list of MCP servers
*   **[Awesome MCP Servers](https://github.com/wong2/awesome-mcp-servers):** Curated list of MCP servers
*   **[Awesome Remote MCP Servers](https://github.com/jaw9c/awesome-remote-mcp-servers):** Curated list of remote MCP servers
*   **[Discord Server](https://glama.ai/mcp/discord):** Community Discord server
*   **[Discord Server (ModelContextProtocol)](https://discord.gg/jHEGxQu2a5):** Community Discord server
*(A full list of community resources is in the original README.)*

## 🤝 Contribute
Contribute to this repository and the MCP ecosystem! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔒 Security
Report security vulnerabilities via [SECURITY.md](SECURITY.md).

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Community
Engage with the MCP community on [GitHub Discussions](https://github.com/orgs/modelcontextprotocol/discussions).

---

*Managed by Anthropic, and built in collaboration with the community. Contribute to the future of LLM integrations!*
```
Key improvements and optimizations in this README:

*   **SEO-Optimized Title and Introduction:**  Includes keywords like "Model Context Protocol," "MCP," "LLM integrations," and uses a hook to grab the reader's attention.
*   **Clear Headings:** Uses proper markdown headings for structure.
*   **Bulleted Key Features:** Highlights the key benefits of using MCP servers.
*   **Concise Summaries:**  Provides brief descriptions of each section.
*   **Actionable Getting Started:** Provides clear, practical instructions for running servers and configuring clients.
*   **Focus on Value:** Emphasizes the benefits of the MCP ecosystem.
*   **Clear Contribution Instructions:**  Directs users to the CONTRIBUTING.md file.
*   **Concise, Informative Text:**  Removes unnecessary phrasing.
*   **Emphasis on Community:**  Highlights the community-driven nature of the project.
*   **Includes Relevant Links:** Provides all necessary links for the user to explore the documentation.
*   **Removed Unnecessary Repetition:** Streamlined the text to avoid repetition.
*   **Concise Lists:** Combined some of the server information to be more concise.
*   **Used the original README's content** and reorganized it to be more easily readable.
*   **Emphasis on value** of the project for end-users.
*   **Removed the "Note" about alphabetical order:** It's more important to focus on clarity than to maintain alphabetical order for its own sake.