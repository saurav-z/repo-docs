<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Seamless AI Integration</h1>

**Unlock the power of AI by effortlessly connecting to 50+ services with Klavis AI's hosted or self-hosted MCP servers.**

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   **🌐 Hosted Service:** Production-ready infrastructure with 99.9% uptime SLA, offering instant access to 50+ integrations without setup.
*   **🐳 Self-Hosted Solutions:** Deploy and customize MCP servers using Docker, allowing for full control and flexibility.
*   **🔐 Enterprise OAuth:** Simplify authentication for Google, GitHub, Slack, and other services.
*   **🛠️ Extensive Integrations:** Connect with CRM tools, productivity platforms, databases, social media, and more.
*   **🚀 Rapid Deployment:** Zero-config setup for tools like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise Ready:** SOC2 compliant, GDPR ready, with dedicated support for enterprise needs.
*   **📖 Open Source:** Benefit from a community-driven project with full source code for customization and self-hosting.

## Quick Start: Deploy MCP Servers in Seconds

Klavis AI makes integrating services with your AI applications simple. Whether you're aiming for speed or control, we have you covered:

### 🌐 Hosted Service (Recommended for Production)

Get up and running in under a minute with our managed infrastructure:

**1.  Get Your API Key:**

```bash
pip install klavis
# or
npm install klavis
```

**2.  Integrate with your code:**
```python
from klavis import Klavis

klavis = Klavis(api_key="your-free-key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

**3.  🔗 [Get Free API Key](https://www.klavis.ai/home/api-keys)**

### 🐳 Docker for Self-Hosting

Easily run servers locally or on your infrastructure.

```bash
# Run GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server-oauth:latest

# Run YouTube MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/youtube-mcp-server:latest
```

### 🖥️ Configuration for Cursor

**For Cursor, use our hosted service URLs directly - no Docker setup needed:**

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

**Generate your personalized configuration in seconds:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select any service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration** for your tool
4.  **Paste into your application's config**

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images (Quickest Way to Start)

Ideal for testing and integrating with AI tools.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
*   `ghcr.io/klavis-ai/{server-name}-mcp-server-oauth:latest` - Server with OAuth support

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server-oauth:latest
```

[**🔗 Get Free API Key →](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Clone and run any MCP server locally or with Docker.

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

**Use Klavis hosted service for rapid integration (no Docker required):**

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers (More Coming Soon!)

| Service        | Docker Image                                                  | OAuth Required | Description                                    |
| -------------- | ------------------------------------------------------------- | -------------- | ---------------------------------------------- |
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`                          | ✅            | Repository management, issues, PRs              |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server-oauth`                     | ✅            | Email reading, sending, management              |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server-oauth`             | ✅            | Spreadsheet operations                         |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`                         | ❌            | Video information, search                     |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server-oauth`                     | ✅            | Channel management, messaging                |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server-oauth`                    | ✅            | Database and page operations                   |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server-oauth`                | ✅            | CRM data management                            |
| **Postgres**   | `ghcr.io/klavis-ai/postgres-mcp-server`                       | ❌            | Database operations                            |
| ...            | ...                                                          | ...            | ...                                            |

**View the full list:**
[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

### **Python**
```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123",
    platform_name="MyApp"
)
```

### **TypeScript**
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

## 🌐 Hosted MCP Service – Effortless AI Integration

Our hosted service offers instant access to a comprehensive suite of MCP servers, streamlining your AI application development.

### ✨ **Why Choose Our Hosted Service:**

*   **🚀 Instant Setup**: Launch any MCP server in seconds.
*   **🔐 OAuth Handled**: We manage complex authentication flows.
*   **🏗️ Zero Infrastructure**: Run your projects on our secure cloud.
*   **📈 Scalability**: Easily scale from prototypes to production.
*   **🔄 Always Updated**: Automatic updates to the latest MCP server versions.
*   **💰 Cost-Effective**: Pay only for the resources you utilize.

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

## 🔐 OAuth Authentication - Simplified

Klavis AI simplifies OAuth implementation for services like Google, GitHub, and Slack.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server-oauth:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```
**Alternative**:
For advanced users, you can implement OAuth yourself by creating apps with each service provider. Check individual server READMEs for technical details.

## 📚 Resources & Community

| Resource           | Link                                            | Description                                    |
| ------------------ | ----------------------------------------------- | ---------------------------------------------- |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)        | Complete guides and API reference                |
| **💬 Discord**     | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users                |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features             |
| **📦 Examples**     | [examples/](examples/)                          | Working examples with popular AI frameworks      |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                   | Individual server documentation                |

## 🤝 Contributing

We welcome contributions!  Help us by:

*   🐛 Reporting bugs or suggesting features
*   📝 Improving documentation
*   🔧 Building new MCP servers
*   🎨 Enhancing existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

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
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repository</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO-Optimized Title & Hook:**  Uses strong keywords like "MCP Servers", "AI Integration", and "Seamless" in the title and a concise introductory sentence to grab attention.
*   **Clear Headings & Structure:**  Organizes information logically with clear headings and subheadings.  This makes it easier to scan and understand.
*   **Bulleted Lists for Key Features:**  Uses bullet points to highlight the core benefits of Klavis AI, making them easy to read and digest.
*   **Actionable Call to Action:**  Includes direct instructions and links to get started (e.g., "Get Free API Key," "Browse All Docker Images").
*   **Emphasis on Benefits:**  Focuses on *what the user gains* from using Klavis AI (e.g., "effortless AI integration", "instant access," "zero setup").
*   **Concise Code Examples:** The code examples are simplified and directly demonstrate how to use Klavis AI.
*   **Detailed Self-Hosting Instructions:**  Provides step-by-step instructions for both Docker and building from source.  The steps are broken down for clarity.
*   **Comprehensive Server Table:**  The table provides a quick overview of available servers, making it easy for users to find what they need.
*   **Clear Explanations:**  Provides explanations for why certain features are important, such as the advantages of the hosted service and the complexities of OAuth.
*   **Strong Community Emphasis:**  Highlights the community resources (Discord, issues, etc.) and encourages contributions.
*   **Link Back to Original Repo:** Added a link back to the original repo in the final section.
*   **Alt text for images:** Added `alt` attributes to the image tags for better accessibility and SEO.
*   **Emphasis on hosted service** The hosted service has been given a higher priority in the structure.
*   **Multiple call to actions:** Included the different ways to contribute to the platform for users interested in different areas.