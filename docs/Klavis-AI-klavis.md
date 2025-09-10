<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Your AI Applications</h1>

<p align="center"><b>Seamlessly integrate with 50+ services using Klavis AI's managed MCP servers or self-host for complete control.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Overview

Klavis AI provides production-ready MCP (Model Connector Protocol) servers, enabling effortless integration of AI applications with various services. Choose between our managed hosted service for ease of use or self-hosting for maximum flexibility. <b><a href="https://github.com/Klavis-AI/klavis">Explore the Klavis AI GitHub repository</a></b>.

## Key Features

*   ✅ **Hosted MCP Service:** Instantly access 50+ pre-built MCP servers with a managed infrastructure.
*   🐳 **Docker Support:** Easily deploy and run MCP servers using Docker images.
*   🔑 **Enterprise-Grade OAuth:** Simplified authentication for services like Google, GitHub, and Slack.
*   ⚙️ **50+ Integrations:** Connect to a wide range of services, including CRM, productivity tools, and social media.
*   🚀 **Instant Deployment:** Seamless setup with popular AI tools like Claude Desktop, VS Code, and Cursor.
*   🛡️ **Enterprise Ready:** SOC2 and GDPR compliant, with dedicated support.
*   📚 **Open Source:** Customizable and self-hostable with full source code access.

## Quick Start: Run Any MCP Server in Minutes

### 🐳 Self-Hosting with Docker

Get your MCP server running quickly with Docker.

1.  **Get a Free API Key (if needed):** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Pull the Docker Image:**
    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    ```
3.  **Run the Server:**
    ```bash
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
    ```
    or (with manual token)
      ```bash
      docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
      ```
4.  **Access the MCP Endpoint:** The server runs on port 5000 and exposes the MCP protocol at `/mcp`.

    Example in Cursor:
    ```json
    {
      "mcpServers": {
        "github": {
          "url": "http://localhost:5000/mcp/"
        }
      }
    }
    ```

### 🌐 Hosted Service (Recommended for Production)

Leverage our managed infrastructure for hassle-free MCP server access.

1.  **Get a Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis Client:**
    ```bash
    pip install klavis  # or npm install klavis
    ```
3.  **Use in Python or Node.js:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

    Example in Cursor:
    ```json
    {
      "mcpServers": {
        "klavis-gmail": {
          "url": "https://gmail-mcp-server.klavis.ai/mcp/?instance_id=your-instance"
        },
        "klavis-github": {
          "url": "https://github-mcp-server.klavis.ai/mcp/?instance_id=your-instance"
        }
      }
    }
    ```

**🚀 Get your personalized configuration instantly:**

1.  **[Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration** for your tool
4.  **Paste into your AI tool** (e.g., Claude Desktop)

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images (Fastest Way)

Pre-built Docker images simplify deployment.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server built from a specific commit ID

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Clone the repository and run MCP servers locally.

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

Detailed instructions are available within each server's README.

## 🛠️ Available MCP Servers

| Service         | Docker Image                                   | OAuth Required | Description                      |
| --------------- | ---------------------------------------------- | -------------- | -------------------------------- |
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`    | ✅             | Email reading, sending, management |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations             |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`        | ❌             | Video information, search        |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`    | ✅             | Channel management, messaging    |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`   | ✅             | Database and page operations     |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management              |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`       | ❌             | Database operations              |
| ...             | ...                                            | ...            | ...                              |

And more!
[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

### Python

```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123"
)
```

### TypeScript

```typescript
import { KlavisClient } from 'klavis';

const klavis = new KlavisClient({ apiKey: 'your-key' });
const server = await klavis.mcpServer.createServerInstance({
    serverName: "Gmail",
    userId: "user123"
});
```

### With AI Frameworks (OpenAI Function Calling)

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

## 🌐 Hosted MCP Service - Zero Setup Required

Benefit from a fully managed service for instant MCP server access.

### ✨ **Why Choose Our Hosted Service:**

*   🚀 **Instant Setup**: Get any MCP server running in seconds.
*   🔐 **OAuth Handled**: Complex authentication simplified.
*   🏗️ **No Infrastructure**: Runs on our secure and scalable cloud.
*   📈 **Auto-Scaling**: Scalable from prototype to production.
*   🔄 **Always Updated**: Latest MCP server versions.
*   💰 **Cost-Effective**: Pay only for what you use, with a free tier.

### 💻 **Quick Integration:**

```python
from klavis import Klavis

klavis = Klavis(api_key="Your-Klavis-API-Key")
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication (For OAuth-Enabled Servers)

Klavis simplifies OAuth, handling the complexities for services requiring authentication.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

Our OAuth wrapper simplifies the complexities.  For advanced users, OAuth can be implemented directly; see individual server READMEs for details.

## 📚 Resources & Community

| Resource          | Link                                       | Description                                 |
| ----------------- | ------------------------------------------ | ------------------------------------------- |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)   | Complete guides and API reference           |
| **💬 Discord**     | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users            |
| **🐛 Issues**      | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)  | Report bugs and request features           |
| **📦 Examples**    | [examples/](examples/)                     | Working examples with popular AI frameworks |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                | Individual server documentation             |

## 🤝 Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>