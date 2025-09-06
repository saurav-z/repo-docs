<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Applications</h1>
<p align="center"><strong>Build smarter AI apps faster with self-hosted or hosted MCP solutions, enterprise-grade OAuth, and 50+ integrations.</strong></p>

<div align="center">
  <a href="https://docs.klavis.ai"><img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation"></a>
  <a href="https://www.klavis.ai"><img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website"></a>
  <a href="https://discord.gg/p7TuTEcssn"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/orgs/klavis-ai/packages"><img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images"></a>
</div>

---

## 🚀 Klavis AI: Simplify AI Integration with MCP Servers

Klavis AI provides a powerful suite of Machine Communication Protocol (MCP) servers, offering both self-hosted and hosted solutions to seamlessly integrate AI applications with various services.  Get started today with a [Free API Key](https://www.klavis.ai/home/api-keys) to unlock the full potential of AI-powered integrations!

**Key Features:**

*   🐳 **Self-Hosted Solutions:** Deploy MCP servers directly within your infrastructure using Docker for maximum control and customization.
*   🌐 **Hosted MCP Service:** Instantly access production-ready MCP servers with a managed infrastructure, eliminating setup and maintenance overhead.
*   🔐 **Enterprise-Grade OAuth:** Simplify authentication for services like Google, GitHub, and Slack, streamlining your integration process.
*   🛠️ **50+ Integrations:** Connect to a wide range of services including CRM, productivity tools, databases, social media, and more.
*   🚀 **Rapid Deployment:** Get up and running with zero-configuration setup for popular AI tools like Claude Desktop, VS Code, and Cursor.

[**View the original repository on GitHub**](https://github.com/Klavis-AI/klavis)

---

## 💡 Quick Start: Run Any MCP Server in 30 Seconds

### 🐳 Self-Hosting with Docker

Quickly deploy MCP servers using Docker for local development or self-managed environments.

**Get your Free API Key to unlock OAuth support:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)

**Examples:**

```bash
# Run Github MCP Server with OAuth Support through Klavis AI
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/github-mcp-server:latest
```

```bash
# Or run GitHub MCP Server (manually add token)
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest
```

*   **Note:** The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

**Example in Cursor:**

```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:5000/mcp/"
    }
  }
}
```

### 🌐 Hosted MCP Service (Recommended for Production)

Experience a production-ready managed infrastructure with over 50 MCP servers, eliminating setup and maintenance.

**Get Started with a Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)

**Quick Integration:**

```bash
pip install klavis
# or
npm install klavis
```

```python
from klavis import Klavis

klavis = Klavis(api_key="Your-Klavis-API-Key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

**Example in Cursor:**

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

**Instant Configuration:**

1.  🔗 [Visit the MCP Servers Page](https://www.klavis.ai/home/mcp-servers)
2.  Select your desired service (Gmail, GitHub, Slack, etc.).
3.  Copy and paste the generated configuration into your tool (e.g., Claude Desktop).

---

## ✨ Key Benefits of the Klavis AI Platform

*   **🌐 Hosted Service**: A fully managed, production-ready infrastructure with a 99.9% uptime SLA.
*   **🔐 Enterprise OAuth**: Seamless and secure authentication across popular platforms (Google, GitHub, Slack, etc.).
*   **🛠️ Extensive Integrations**: Access to 50+ pre-built integrations for a wide array of CRM, productivity, database, and social media tools.
*   **🚀 Instant Deployment**: Effortless setup for AI tools like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise-Ready**: SOC2 and GDPR compliant, with dedicated support for enterprise deployments.
*   **📖 Open Source**:  Customize and self-host with full access to the source code.

---

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images (Fastest Method)

Ideal for testing and integrating with AI tools like Claude Desktop.

**Available Images:**
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Build by selected commit ID

[**🔍 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Build and run any MCP server locally.

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

*   **Note:** Individual server READMEs contain detailed setup instructions.

### Hosted Service for Simplification

**Use our managed infrastructure, eliminating the need for Docker:**

```bash
pip install klavis  # or npm install klavis
```

---

## 🛠️ Available MCP Servers

| Service | Docker Image | OAuth Required | Description |
|---------|--------------|----------------|-------------|
| **GitHub** | `ghcr.io/klavis-ai/github-mcp-server` | ✅ | Repository management, issues, PRs |
| **Gmail** | `ghcr.io/klavis-ai/gmail-mcp-server:latest` | ✅ | Email reading, sending, management |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅ | Spreadsheet operations |
| **YouTube** | `ghcr.io/klavis-ai/youtube-mcp-server` | ❌ | Video information, search |
| **Slack** | `ghcr.io/klavis-ai/slack-mcp-server:latest` | ✅ | Channel management, messaging |
| **Notion** | `ghcr.io/klavis-ai/notion-mcp-server:latest` | ✅ | Database and page operations |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅ | CRM data management |
| **Postgres** | `ghcr.io/klavis-ai/postgres-mcp-server` | ❌ | Database operations |
| ... | ... | ...| ... |

And more!
[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

---

## 💡 Usage Examples

**Leverage existing MCP implementations:**

**Python**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123"
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

### Integrate with AI Frameworks

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

---

## 🌐 Hosted MCP Service - Zero Setup Required

**Ideal for instant access without complex infrastructure management.**

### ✨ Why Choose the Hosted Service:

*   **🚀 Instant Setup**: Launch any MCP server in seconds.
*   **🔐 OAuth Simplified**: No complex authentication setup required.
*   🏗️ **No Infrastructure**: All operations run on our secure, scalable cloud.
*   **📈 Auto-Scaling**: Scale seamlessly from prototyping to production.
*   **🔄 Always Updated**: Access the latest MCP server versions automatically.
*   **💰 Cost-Effective**: Pay only for what you utilize, including a free tier.

### 💻 Quick Integration

```python
from klavis import Klavis

# Get started with just an API key
klavis = Klavis(api_key="Your-Klavis-API-Key")

# Create any MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

# Server is ready to use immediately
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

---

## 🔐 OAuth Authentication (For OAuth-Enabled Servers)

Effortlessly manage complex OAuth authentication for services like Google, GitHub, and Slack.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Simplified OAuth Implementation:**

*   🔧 **Complex Setup**:  OAuth app creation, redirect URLs, scopes, and credential setup are handled automatically.
*   📝 **Implementation Overhead**:  Token refreshing and secure storage are managed seamlessly.
*   🔑 **Credential Management**: No need to manage multiple OAuth app secrets.
*   🔄 **Token Lifecycle**:  Token expiration and refresh processes are handled.

For advanced users, implement OAuth using the individual server READMEs for technical guidance.

---

## 📚 Resources & Community

| Resource             | Link                                     | Description                                 |
| -------------------- | ---------------------------------------- | ------------------------------------------- |
| **📖 Documentation**  | [docs.klavis.ai](https://docs.klavis.ai) | Comprehensive guides and API reference     |
| **💬 Discord**        | [Join Community](https://discord.gg/p7TuTEcssn) | Get support and connect with other users  |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request new features      |
| **📦 Examples**      | [examples/](examples/)                   | Working examples using popular AI frameworks |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)            | Individual server documentation             |

---

## 🤝 Contributing

We welcome contributions!  Contribute by:

*   🐛 Reporting bugs or requesting new features.
*   📝 Improving documentation.
*   🔧 Building new MCP servers.
*   🎨 Enhancing existing servers.

See our [Contributing Guide](CONTRIBUTING.md) to get started.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for more details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications Today!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>