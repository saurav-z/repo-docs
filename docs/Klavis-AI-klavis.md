<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80">
  </picture>
</div>

<h1 align="center">Klavis AI: The Production-Ready MCP Server Solution</h1>
<p align="center"><strong>Unlock seamless integration of AI tools with Klavis, offering both hosted and self-hosted MCP servers.</strong></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## 🔑 Key Features

*   **🌐 Hosted Service:** Production-ready, managed infrastructure with 99.9% uptime, ideal for quick integration.
*   **🐳 Self-Hosted Solutions:** Docker-based and source code options for flexible deployment and customization.
*   **🔐 Enterprise OAuth:** Secure authentication for Google, GitHub, Slack, and other services.
*   **🛠️ Extensive Integrations:** Access 50+ MCP servers for CRM, productivity tools, databases, and more.
*   **🚀 Rapid Deployment:** Integrate with tools like Claude Desktop, VS Code, and Cursor with zero-configuration.
*   **🏢 Enterprise-Ready:** SOC2 and GDPR compliant, with dedicated support.
*   **📖 Open Source:** Customize and self-host with fully available source code.

## 🚀 Quick Start: Get Your MCP Server Up and Running

Whether you prefer a hosted solution or self-hosting, Klavis AI offers quick setup options.

### 1. 🌐 Hosted Service (Recommended)

Leverage our managed infrastructure to access over 50 MCP servers without setup.

*   **Benefits:** Instant access, handled OAuth, no infrastructure management, auto-scaling, and constant updates.
*   **Quick Integration:**

```python
from klavis import Klavis

# Get started with your API key
klavis = Klavis(api_key="your-free-key")

# Create any MCP server
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id",
    platform_name="MyApp"
)

# Server is ready to use
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

*   **Get Started:** **🔗 [Get Free API Key](https://www.klavis.ai/home/api-keys)**

### 2. 🐳 Docker (Self-Hosting)

Quickly set up MCP servers using Docker for local development and integration.

```bash
# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 3. 🖥️ Cursor Configuration

**Configure Klavis MCP Servers in Cursor:**

1.  **🔗 [Visit MCP Servers Page](https://www.klavis.ai/home/mcp-servers)**.
2.  Select your desired service (Gmail, GitHub, etc.).
3.  Copy the generated configuration.
4.  Paste into your Cursor configuration.

## 🛠️ Available MCP Servers

Klavis AI offers a growing list of MCP servers to connect your AI applications.

| Service          | Docker Image                                       | OAuth Required | Description                      |
| ---------------- | -------------------------------------------------- | -------------- | -------------------------------- |
| GitHub           | `ghcr.io/klavis-ai/github-mcp-server`              | ✅             | Repository management, issues, PRs |
| Gmail            | `ghcr.io/klavis-ai/gmail-mcp-server:latest`         | ✅             | Email reading, sending, management |
| Google Sheets    | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations             |
| YouTube          | `ghcr.io/klavis-ai/youtube-mcp-server`             | ❌             | Video information, search        |
| Slack            | `ghcr.io/klavis-ai/slack-mcp-server:latest`         | ✅             | Channel management, messaging    |
| Notion           | `ghcr.io/klavis-ai/notion-mcp-server:latest`        | ✅             | Database and page operations     |
| Salesforce       | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`    | ✅             | CRM data management              |
| Postgres         | `ghcr.io/klavis-ai/postgres-mcp-server`            | ❌             | Database operations              |
| ...              | ...                                                | ...            | ...                              |

**Explore all 50+ servers:**

*   **🔍 [View All Servers](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)**
*   **🐳 [Browse Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)**

## 🎯 Self-Hosting Instructions

Detailed instructions to get started with self-hosting Klavis AI servers.

### 1.  🐳 Docker Images

For quick and easy deployment, use our pre-built Docker images.

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Build and run MCP servers locally for customization.

```bash
git clone https://github.com/klavis-ai/klavis.git
cd klavis/mcp_servers/github

# Option A: Using Docker
docker build -t github-mcp .
docker run -p 5000:5000 github-mcp

# Option B: Run directly (Go example)
go mod download
go run server.go

# Option C: Python servers
cd ../youtube
pip install -r requirements.txt
python server.py

# Option D: Node.js servers
cd ../slack
npm install
npm start
```

### With AI Frameworks

**Python**
```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123",
    platform_name="MyApp"
)
```

**TypeScript**
```typescript
import { KlavisClient } from 'klavis';

const klavis = new KlavisClient({ apiKey: 'your-key' });
const server = await klavis.mcpServer.createServerInstance({
    serverName: "Gmail",
    userId: "user123"
});
```

**OpenAI Function Calling**
```python
from openai import OpenAI
from klavis import Klavis

klavis = Klavis(api_key="your-key")
openai = OpenAI(api_key="your-openai-key")

# Create server and get tools
server = klavis.mcp_server.create_server_instance("YOUTUBE", "user123")
tools = klavis.mcp_server.list_tools(server.server_url, format="OPENAI")

# Use with OpenAI
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this video: https://..."}],
    tools=tools.tools
)
```

[**📖 View Complete Examples →**](examples/)

## 🔐 OAuth Authentication

Klavis AI simplifies OAuth integration, handling the complexities.  The hosted service includes built-in OAuth for ease of use, while self-hosting provides flexibility.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

## 📚 Resources & Community

| Resource         | Link                                                            | Description                     |
| ---------------- | --------------------------------------------------------------- | ------------------------------- |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)                        | Complete guides and API reference |
| **💬 Discord**    | [Join Community](https://discord.gg/p7TuTEcssn)                  | Get help and connect with users |
| **🐛 Issues**      | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)      | Report bugs and request features |
| **📦 Examples**    | [examples/](examples/)                                          | Working examples with AI frameworks |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                                   | Individual server documentation   |

## 🤝 Contributing

Contribute to Klavis AI!  We welcome contributions in the form of:

*   🐛 Bug reports and feature requests.
*   📝 Documentation improvements.
*   🔧 New MCP server development.
*   🎨 Improvements to existing servers.

See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repo</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  The title and first sentence now use keywords like "MCP Server," "AI," "Integration," and "Hosted" to improve search ranking.  The use of headings also helps with SEO.
*   **Concise Hook:** The opening sentence and overall structure is improved, immediately telling the user what Klavis does and its benefits.
*   **Clear Headings and Structure:** The README is now clearly organized with headings and subheadings for easy navigation.
*   **Bulleted Key Features:** Key features are bulleted for readability and emphasis.
*   **Emphasis on Benefits:** The "Hosted Service" section clearly outlines the advantages.
*   **Concise and Actionable Instructions:** The "Quick Start" and "Self-Hosting" sections provide clear, actionable steps.
*   **Complete Examples:** Examples are provided for common use cases.
*   **Call to Action:**  Consistent calls to action (e.g., "Get Free API Key," "View All Servers") are included.
*   **GitHub Repo Link Back:** The final call to action includes a direct link back to the original GitHub repository.
*   **Internal Links:** Uses internal links within the document (like for the `Get Free API Key`) to improve SEO and UX.
*   **Keyword Rich:** Uses relevant keywords naturally throughout the README, like "Docker," "OAuth," and specific service names (GitHub, Gmail, etc.).
*   **Improved Formatting:**  Uses consistent formatting (bolding, code blocks, etc.) for better readability.
*   **Removed Redundancy:** Streamlined some sections to remove unnecessary information.
*   **Focus on User Value:**  The README focuses on what users can *do* with Klavis and the benefits they receive.