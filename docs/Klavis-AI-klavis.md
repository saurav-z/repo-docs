<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unlock AI Application Potential with Production-Ready MCP Servers</h1>

<p align="center">**Simplify AI integration with ready-to-use MCP servers for various services, including Gmail, GitHub, and more.  Get instant access with a hosted solution or self-host for complete control.**</p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   **🚀 Instant Integration:**  Quickly connect to over 50+ services like Gmail, GitHub, and more.
*   **🌐 Hosted Service:** Production-ready, managed infrastructure with 99.9% uptime.  Focus on your application, not infrastructure!
*   **🐳 Self-Hosting Options:** Deploy using Docker for full control and customization.
*   **🔐 Enterprise-Grade Security:**  Supports Enterprise OAuth for secure authentication, including Google, GitHub, and Slack.
*   **🛠️ Extensive Integrations:** Compatible with numerous CRM, productivity tools, databases, and social media platforms.
*   **📦 Ready for AI Frameworks:** Seamlessly integrates with OpenAI and other AI tools via function calling.
*   **💡 Easy Configuration:**  Simple setup for Claude Desktop, VS Code, Cursor, and other AI-powered applications.
*   **🏢 Enterprise-Ready:** SOC2 compliant and GDPR ready.
*   **📖 Open Source:**  Full source code available for customization and self-hosting (see [Klavis on GitHub](https://github.com/Klavis-AI/klavis)).

## Quick Start: Get Started in Seconds

### 1.  **🌐 Hosted Service (Recommended)**

Get up and running instantly with our managed infrastructure. Ideal for production environments.

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)**

```bash
pip install klavis
# or
npm install klavis
```

```python
from klavis import Klavis

klavis = Klavis(api_key="your-free-key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

### 2.  🐳 **Docker (Self-Hosting)**

Self-host your MCP servers for greater control.

```bash
# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 3. 🖥️ **Cursor Configuration**

Integrate directly with Cursor using our hosted service.

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

**Get your configuration:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  Select a service (Gmail, GitHub, etc.).
3.  Copy the generated configuration.
4.  Paste into your AI tool config (e.g., Claude Desktop).

## Enterprise-Grade Infrastructure

Klavis AI offers a robust platform for integrating MCP servers into your AI applications, designed for:

*   **Production Ready:**  Reliable hosted service with a 99.9% uptime SLA.
*   **Secure Authentication:** Built-in support for Enterprise OAuth with services like Google, GitHub, and Slack.
*   **Broad Compatibility:** Seamlessly integrate with 50+ services, including popular CRM, productivity, and social media platforms.
*   **Rapid Deployment:**  Easily set up and connect MCP servers with popular AI tools like Claude Desktop, VS Code, and Cursor.
*   **Compliance and Support:** Designed with Enterprise needs in mind with SOC2 compliance, GDPR readiness, and dedicated support.
*   **Open Source Flexibility:** Access to the complete source code enables customization and self-hosting options.

## 🎯 Self-Hosting Instructions

### 1.  🐳 Docker Images

The fastest way to start.  Perfect for testing or integrating with AI tools.

**Available Images:**
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth support

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Clone the repository and run servers locally, with or without Docker.

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

Each server directory contains detailed setup instructions.

To use with our managed infrastructure:

```bash
pip install klavis  # or npm install klavis
```

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

## 💡 Usage Examples

For existing MCP implementations:

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

### With AI Frameworks

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

## 🌐 Hosted MCP Service - Zero Setup Required

**Perfect for instant access without infrastructure concerns.**

### ✨ **Benefits:**

*   **🚀 Instant Setup:** Get any MCP server running within 30 seconds.
*   **🔐 OAuth Handled:** Simplified authentication setup.
*   **🏗️ No Infrastructure Management:** Operates on our secure, scalable cloud.
*   **📈 Auto-Scaling:** Seamlessly scales from prototype to production.
*   **🔄 Automatic Updates:** Always running the latest versions of MCP servers.
*   **💰 Cost-Effective:** Pay only for the resources you consume, with a free tier available.

### 💻 **Quick Integration:**

```python
from klavis import Klavis

# Get started with just an API key
klavis = Klavis(api_key="your-free-key")

# Create any MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id",
    platform_name="MyApp"
)

# Server is ready to use immediately
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication

Simplify complex OAuth integration.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Benefits of using Klavis's OAuth Implementation:**

*   **Simplified Setup:** Handles the complexities of creating OAuth apps.
*   **Token Management:** Secure token storage and refresh mechanisms are provided.
*   **Reduced Overhead:** Allows you to focus on utilizing MCP servers.

**Alternative Option:** Advanced users can manage OAuth independently; consult the specific server's README files for details.

## 📚 Resources & Community

| Resource | Link | Description |
|----------|------|-------------|
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai) | Complete guides and API reference |
| **💬 Discord** | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users |
| **🐛 Issues** | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features |
| **📦 Examples** | [examples/](examples/) | Working examples with popular AI frameworks |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/) | Individual server documentation |

## 🤝 Contributing

Contributions are welcome! We appreciate contributions in the form of:

*   🐛 Bug reports and feature requests.
*   📝 Documentation improvements.
*   🔧 New MCP server development.
*   🎨 Enhancements to existing servers.

Consult the [Contributing Guide](CONTRIBUTING.md) to start contributing.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">View on GitHub</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:** Included keywords like "MCP servers," "AI integration," "OAuth," "self-hosting," and service names (Gmail, GitHub, etc.)  in headings and content.
*   **One-Sentence Hook:**  The opening paragraph provides a clear, compelling summary of what Klavis AI offers.
*   **Clear Headings and Structure:**  Used `h1`, `h2`, and `h3` tags to create a well-organized and scannable document.
*   **Bulleted Key Features:**  Made it easy to quickly grasp the benefits of using Klavis AI.
*   **Action-Oriented Language:**  Used strong verbs (e.g., "Unlock," "Simplify," "Get Started") to encourage engagement.
*   **Concise Explanations:**  Simplified explanations to avoid overwhelming users.
*   **Call to Action:**  Included clear calls to action (e.g., "Get Free API Key," "View All 50+ Servers").
*   **Emphasis on Benefits:** Focused on *what* Klavis AI helps users achieve (e.g., "supercharge AI applications").
*   **Clear Differentiation:** Highlighted the advantages of the hosted service vs. self-hosting.
*   **GitHub Link:** Added a direct link back to the original GitHub repository.
*   **Alt Text:** Added `alt` text to the image tag.
*   **Removed Redundancy:** Streamlined the content by removing repeated information.
*   **Improved Readability:**  Used spacing and formatting to improve readability.
*   **Included a GitHub link under the final CTA.**