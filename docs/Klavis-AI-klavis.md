<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Integration</h1>
<p align="center"><strong>Easily integrate your AI applications with over 50+ services using Klavis AI's managed and self-hosted MCP servers.</strong></p>

<div align="center">
  <a href="https://docs.klavis.ai"><img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation"></a>
  <a href="https://www.klavis.ai"><img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website"></a>
  <a href="https://discord.gg/p7TuTEcssn"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/orgs/klavis-ai/packages"><img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images"></a>
</div>

## Key Features

*   🌐 **Hosted Service**: Production-ready, managed infrastructure with a 99.9% uptime SLA.
*   🔐 **Enterprise OAuth**: Seamless authentication for Google, GitHub, Slack, Salesforce, and more.
*   🛠️ **50+ Integrations**:  Connect to CRM, productivity tools, databases, social media, and beyond.
*   🚀 **Instant Deployment**: Zero-config setup for popular tools like Claude Desktop, VS Code, and Cursor.
*   🏢 **Enterprise-Ready**:  SOC2 compliant, GDPR ready, with dedicated support.
*   🐳 **Self-Hosting Options**: Docker images and source code available for customization.
*   📖 **Open Source**: Full source code available for customization and self-hosting.

## 🚀 Quick Start: Integrate AI with Services in Seconds

Klavis AI provides multiple ways to quickly get your AI applications connected to a multitude of services.

### 🌐 Hosted Service (Recommended for Production)

Leverage our fully managed infrastructure for instant access to 50+ MCP servers with no setup.

**Get Started:**

1.  **🔗 [Get Free API Key](https://www.klavis.ai/home/api-keys)**

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

### 🐳 Docker (Self-Hosting)

For local testing or custom integration, self-host with Docker.

```bash
# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 🖥️ Cursor Configuration

Integrate directly with Cursor using our hosted service URLs.  No Docker setup is needed.

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

**Get Personalized Configuration:**

1.  **🔗 [Visit our MCP Servers Page](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.).
3.  **Copy the generated configuration** and paste it into your tool's config.

## 🎯 Self-Hosting Guide

Choose your preferred method for self-hosting.

### 1. 🐳 Docker Images

Easily deploy MCP servers using Docker images.

**Available Images:**
* `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
* `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth support

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

For advanced users, build and run servers from source code.

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

## 🛠️ Available MCP Servers

| Service           | Docker Image                                       | OAuth Required | Description                                   |
| ----------------- | -------------------------------------------------- | -------------- | --------------------------------------------- |
| **GitHub**        | `ghcr.io/klavis-ai/github-mcp-server`               | ✅             | Repository management, issues, PRs           |
| **Gmail**         | `ghcr.io/klavis-ai/gmail-mcp-server:latest`         | ✅             | Email reading, sending, management         |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                     |
| **YouTube**       | `ghcr.io/klavis-ai/youtube-mcp-server`              | ❌             | Video information, search                    |
| **Slack**         | `ghcr.io/klavis-ai/slack-mcp-server:latest`         | ✅             | Channel management, messaging              |
| **Notion**        | `ghcr.io/klavis-ai/notion-mcp-server:latest`        | ✅             | Database and page operations               |
| **Salesforce**    | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`    | ✅             | CRM data management                      |
| **Postgres**      | `ghcr.io/klavis-ai/postgres-mcp-server`             | ❌             | Database operations                        |
| ...               | ...                                                | ...            | ...                                           |

**[🔍 View All 50+ Servers →](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)** | **[🐳 Browse Docker Images →](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)**

## 💡 Usage Examples

Here are example snippets to get started, assuming you have an API key from our hosted service.

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

### Integrating with AI Frameworks

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

**[📖 View Complete Examples →](examples/)**

## 🌐 Hosted MCP Service: Simplified Integration

Our hosted service eliminates infrastructure complexity, perfect for individuals and businesses.

### ✨ **Benefits of Our Hosted Service:**

*   🚀 **Instant Setup**:  Get any MCP server running in seconds.
*   🔐 **OAuth Handled**:  Simplify authentication setup.
*   🏗️ **No Infrastructure**:  Leverage our secure, scalable cloud.
*   📈 **Auto-Scaling**:  Scale seamlessly from prototype to production.
*   🔄 **Always Updated**:  Benefit from the latest MCP server versions.
*   💰 **Cost-Effective**: Pay only for what you use, with a free tier available.

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

Some services (like Google, GitHub, Slack) require OAuth. Klavis AI simplifies this process.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth Needs Additional Implementation**
- 🔧 **Complex Setup**: Creating OAuth apps and handling redirect URLs
- 📝 **Implementation Overhead**: Handling OAuth 2.0 flow, token refresh and more
- 🔑 **Credential Management**: Managing multiple secrets
- 🔄 **Token Lifecycle**: Dealing with expirations and refresh errors.

Klavis handles the complexity, so you can focus on using the servers.

**Alternative:** Advanced users can implement OAuth directly - check individual server READMEs for details.

## 📚 Resources & Community

| Resource               | Link                                             | Description                                      |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------ |
| **📖 Documentation**   | [docs.klavis.ai](https://docs.klavis.ai)         | Complete guides and API reference                |
| **💬 Discord**         | [Join Community](https://discord.gg/p7TuTEcssn)  | Get help, connect with users, and ask questions. |
| **🐛 Issues**          | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features                 |
| **📦 Examples**        | [examples/](examples/)                           | Working examples with popular AI frameworks       |
| **🔧 Server Guides**  | [mcp_servers/](mcp_servers/)                     | Individual server documentation                  |

## 🤝 Contributing

We welcome contributions!  Please check out our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repository</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:** Added keyword-rich titles (e.g., "Production-Ready MCP Servers," "AI Integration") and phrases.  The repeated use of "MCP Server" is critical.
*   **One-Sentence Hook:**  Provides an immediate value proposition for the product.
*   **Clear Headings:**  Organizes the content for readability and SEO.
*   **Bulleted Key Features:**  Highlights key benefits.
*   **Emphasis on Value:** Focuses on *what* the product does for the user (integrating with services) rather than just *what* the product *is*.
*   **Actionable CTAs:**  Clear calls to action (e.g., "Get Free API Key," "Join Community").
*   **Code Snippets with Context:**  Includes relevant code examples, making it easy to understand and use.
*   **Comprehensive Guides:** Guides the user step-by-step through important actions.
*   **Multiple Integration Paths:** Addresses both hosted and self-hosted users.
*   **Community Engagement:** Directs users to documentation, Discord, and examples.
*   **Improved Formatting:**  Consistent use of Markdown for better readability and SEO benefits.
*   **Alt Text for Images:**  Adds alt text to images for accessibility and SEO.
*   **Direct Link Back to the Repo:** Added a final CTA to the repository.
*   **Includes Key Terms:**  Repeats important keywords like "MCP Server", "AI Integration"

This revised README is much more user-friendly, SEO-optimized, and provides a strong introduction to Klavis AI's offerings, encouraging exploration and adoption.