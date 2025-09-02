<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Applications</h1>

<p align="center"><b>Unlock seamless integration with 50+ MCP servers, empowering your AI applications with Klavis AI.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   **🌐 Hosted Service**:  Managed infrastructure with 99.9% uptime SLA for instant access to 50+ MCP servers.
*   **🐳 Self-Hosted Solutions**: Deploy and customize MCP servers using Docker and source code.
*   **🔐 Enterprise OAuth**:  Secure and streamlined authentication for popular services like Google, GitHub, and Slack.
*   **🛠️ Extensive Integrations**: Connect to CRM, productivity tools, databases, social media, and more.
*   **🚀 Rapid Deployment**:  Integrate quickly with platforms like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise-Grade Security**: SOC2 and GDPR compliant.
*   **📖 Open Source**:  Full source code available for flexibility and customization.

##  🚀 Quick Start: Run Any MCP Server in 30 Seconds

Klavis AI simplifies the integration of AI tools with various services.  Choose your preferred deployment method:

### 1. 🌐 Hosted Service (Recommended)

Get instant access to 50+ MCP servers with our managed infrastructure.  No setup is required.

[Get Free API Key →](https://www.klavis.ai/home/api-keys)

**Installation (Python):**

```bash
pip install klavis
```

**Installation (JavaScript):**

```bash
npm install klavis
```

**Example (Python):**

```python
from klavis import Klavis

klavis = Klavis(api_key="Your-Klavis-API-Key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

**Example (JSON - for tools like Cursor):**

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

### 2. 🐳 Self-Hosting with Docker

Deploy MCP servers with ease using Docker containers.

[Get Free API Key →](https://www.klavis.ai/home/api-keys) (Required for OAuth-enabled servers)

**Example: Running a GitHub MCP Server (with OAuth support):**

```bash
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/github-mcp-server:latest
```

**Example: Running a GitHub MCP Server (manual token):**

```bash
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest
```

**Example (JSON - for tools like Cursor):**

```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:5000/mcp/"
    }
  }
}
```

**Personalized Configuration:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.).
3.  **Copy the generated configuration** for your AI tool.
4.  **Paste** the configuration into your AI application.

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images

The fastest way to get started.  Ideal for testing or integrating with AI tools.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` (basic server)
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` (with OAuth support)

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

**Examples:**

```bash
# GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Clone the repository and run any MCP server locally.

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

Detailed setup instructions are available in each server's `README`.

You can also use our managed infrastructure to avoid Docker:

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers

| Service          | Docker Image                                     | OAuth Required | Description                                         |
| ---------------- | ------------------------------------------------ | -------------- | --------------------------------------------------- |
| **GitHub**       | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs                  |
| **Gmail**        | `ghcr.io/klavis-ai/gmail-mcp-server:latest`      | ✅             | Email reading, sending, management                  |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                              |
| **YouTube**      | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌             | Video information, search                           |
| **Slack**        | `ghcr.io/klavis-ai/slack-mcp-server:latest`      | ✅             | Channel management, messaging                      |
| **Notion**       | `ghcr.io/klavis-ai/notion-mcp-server:latest`     | ✅             | Database and page operations                      |
| **Salesforce**   | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                               |
| **Postgres**     | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌             | Database operations                               |
| ...              | ...                                              | ...            | ...                                                 |

And many more!
[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

**Python:**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123"
)
```

**TypeScript:**

```typescript
import { KlavisClient } from 'klavis';

const klavis = new KlavisClient({ apiKey: 'your-key' });
const server = await klavis.mcpServer.createServerInstance({
    serverName: "Gmail",
    userId: "user123"
});
```

### With AI Frameworks

**OpenAI Function Calling:**

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

**Ideal for individuals and businesses seeking quick access without infrastructure overhead:**

### ✨ Why Choose Our Hosted Service:

*   **🚀 Instant Setup**: Get any MCP server running in 30 seconds.
*   **🔐 OAuth Simplified**: No complex authentication setup needed.
*   **🏗️ No Infrastructure**: Operate on our secure, scalable cloud.
*   **📈 Automated Scaling**: Seamlessly scale from prototype to production.
*   **🔄 Automatic Updates**: Always have the latest MCP server versions.
*   **💰 Cost-Effective**: Pay only for the resources you consume, with a free tier available.

### 💻 Quick Integration:

```python
from klavis import Klavis

# Get started with an API key
klavis = Klavis(api_key="Your-Klavis-API-Key")

# Create an MCP server immediately
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

# Server is ready to use
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication

Some servers require OAuth for authentication (Google, GitHub, Slack, etc.).

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**OAuth Implementation Challenges:**

*   **🔧 Complex Setup**: Creating OAuth apps, managing redirects, scopes, and credentials.
*   **📝 Implementation Overhead**: Handling OAuth 2.0 flows, callbacks, token refreshes, and secure storage.
*   🔑 **Credential Management**: Managing multiple OAuth app secrets across different services.
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases

Klavis AI simplifies OAuth by managing these complexities, so you can directly utilize MCP servers.

**Alternative**: Advanced users can implement OAuth by creating apps with service providers. See server-specific `README` files for details.

## 📚 Resources & Community

| Resource                 | Link                                                | Description                                   |
| ------------------------ | --------------------------------------------------- | --------------------------------------------- |
| **📖 Documentation**      | [docs.klavis.ai](https://docs.klavis.ai)             | Comprehensive guides and API reference        |
| **💬 Discord**           | [Join Community](https://discord.gg/p7TuTEcssn)     | Get help and connect with other users       |
| **🐛 Issues**            | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features            |
| **📦 Examples**          | [examples/](examples/)                             | Working examples with popular AI frameworks |
| **🔧 Server Guides**      | [mcp_servers/](mcp_servers/)                        | Individual server documentation               |

## 🤝 Contributing

We welcome contributions!  Help us by:

*   🐛 Reporting bugs or requesting new features.
*   📝 Improving the documentation.
*   🔧 Building new MCP servers.
*   🎨 Enhancing existing servers.

Check out our [Contributing Guide](CONTRIBUTING.md) to start contributing.

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
Key improvements and SEO optimizations:

*   **Clear Title & Hook:** A concise, SEO-friendly title and a one-sentence hook to grab attention.
*   **Structured Headings:**  Organized content with clear headings and subheadings for readability and SEO benefits.
*   **Bulleted Key Features:**  Uses bullet points to highlight key benefits and features.
*   **Keyword Optimization:**  Incorporates relevant keywords like "MCP servers," "AI applications," "OAuth," "Docker," and specific service names (Gmail, GitHub, etc.) naturally throughout the text.
*   **Calls to Action:**  Prominent calls to action with links (Get Free API Key, Documentation, etc.).
*   **Code Examples:**  Includes relevant code examples with clear explanations.
*   **Concise Language:**  Uses clear and concise language to convey information efficiently.
*   **Links Back to Repo:** Includes a final link back to the original GitHub repository for attribution.
*   **Alt Text for Images:** Adds alt text to images for accessibility and SEO.
*   **Comprehensive Overview:**  Provides a complete overview of the project's features and usage.
*   **Updated and Complete:** Includes all the original sections, organized and made clear.