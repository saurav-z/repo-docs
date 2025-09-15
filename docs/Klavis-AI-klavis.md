<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI</h1>

<p align="center"><b>Unlock the power of 50+ MCP Servers with Klavis AI, offering both self-hosted and managed solutions for effortless AI integration.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## 🚀 **Klavis AI: Seamlessly Integrate AI with Your Favorite Tools**

Klavis AI provides production-ready MCP (Model Control Protocol) servers, offering both self-hosted and managed solutions to connect your AI applications with various services.  From instant deployment to enterprise-grade security, Klavis AI simplifies AI integration.  

**[Explore the Klavis AI GitHub Repository](https://github.com/Klavis-AI/klavis)**

**Key Features:**

*   🐳 **Self-Hosted Solutions:** Deploy MCP servers quickly using Docker, giving you full control.
*   🌐 **Hosted MCP Service:**  Managed infrastructure with 99.9% uptime for rapid, zero-configuration deployment.
*   🔐 **Enterprise OAuth:** Secure and seamless authentication for services like Google, GitHub, and Slack.
*   🛠️ **50+ Integrations:** Connect to CRMs, productivity tools, databases, social media, and more.
*   🚀 **Instant Deployment:** Get up and running in seconds with support for Claude Desktop, VS Code, and Cursor.
*   🏢 **Enterprise-Ready:** Compliant with SOC2 and GDPR, with dedicated support for your needs.

## ⚙️ **Getting Started: Quick Setup Options**

### 🐳 **1. Self-Hosting with Docker**

Easily run MCP servers in your environment.  Get started in 30 seconds.

**To run a GitHub MCP Server:**

1.  **Get your Free API Key**:  [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Pull the Docker Image:**

    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    ```

3.  **Run the Container (with OAuth):**

    ```bash
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
    ```

4.  **Or Run the Container (manual token):**

    ```bash
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
    ```

**Note:**  The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

**Example usage in Cursor:**
```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:5000/mcp/"
    }
  }
}
```

### 🌐 **2. Hosted Service (Recommended)**

For production environments, our managed service provides instant access to 50+ MCP servers with no setup required.

1.  **Get your Free API Key**:  [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis Library:**

    ```bash
    pip install klavis
    # or
    npm install klavis
    ```

3.  **Integrate with Python**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

4.  **Integrate with Javascript**

    ```javascript
    import { KlavisClient } from 'klavis';

    const klavis = new KlavisClient({ apiKey: 'your-key' });
    const server = await klavis.mcpServer.createServerInstance({
        serverName: "Gmail",
        userId: "user123"
    });
    ```

**Example usage in Cursor:**

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

**💡 Get your personalized configuration instantly:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select any service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration** for your tool
4.  **Paste into Claude Desktop config** - done!

## ✨ **Enterprise-Grade MCP Infrastructure**

Klavis AI offers a robust and reliable platform for all your MCP server needs:

*   **🌐 Hosted Service**: Managed infrastructure with a 99.9% uptime SLA.
*   **🔐 Enterprise OAuth**: Simplified authentication for all major services.
*   🛠️ **50+ Integrations**: Extensive support for CRM, productivity, databases, social media, and more.
*   🚀 **Instant Deployment**: Seamless setup for AI tools like Claude Desktop, VS Code, and Cursor.
*   🏢 **Enterprise Ready**: SOC2 and GDPR compliant.
*   📖 **Open Source**:  Customize and self-host with full source code availability.

## 🎯 **Self-Hosting Instructions in Depth**

### 🐳 **1. Docker Images (Fastest Way to Start)**

Perfect for quickly testing or integrating with AI tools like Claude Desktop.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Build by selected commit ID

**Browse Docker Images:** [**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 🏗️ **2. Build from Source**

Clone and run any MCP server locally (with or without Docker):

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

Each server includes detailed setup instructions in its individual README.

**Alternatively, utilize our managed infrastructure - no Docker required:**

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ **Available MCP Servers**

| Service           | Docker Image                                        | OAuth Required | Description                                      |
| ----------------- | --------------------------------------------------- | -------------- | ------------------------------------------------ |
| **GitHub**        | `ghcr.io/klavis-ai/github-mcp-server`               | ✅             | Repository management, issues, PRs              |
| **Gmail**         | `ghcr.io/klavis-ai/gmail-mcp-server:latest`        | ✅             | Email reading, sending, management              |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                          |
| **YouTube**       | `ghcr.io/klavis-ai/youtube-mcp-server`              | ❌             | Video information, search                       |
| **Slack**         | `ghcr.io/klavis-ai/slack-mcp-server:latest`         | ✅             | Channel management, messaging                   |
| **Notion**        | `ghcr.io/klavis-ai/notion-mcp-server:latest`        | ✅             | Database and page operations                    |
| **Salesforce**    | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`    | ✅             | CRM data management                             |
| **Postgres**      | `ghcr.io/klavis-ai/postgres-mcp-server`             | ❌             | Database operations                             |
| ...               | ...                                                 | ...            | ...                                              |

And many more!

[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 **Usage Examples**

For existing MCP implementations:

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

## 🌐 **Hosted MCP Service - Zero Setup Required**

**Perfect for individuals and businesses seeking instant access without the complexities of managing infrastructure:**

### ✨ **Why Choose Our Hosted Service:**

*   🚀 **Instant Setup**: Launch any MCP server within seconds.
*   🔐 **OAuth Handled**: Eliminate complex authentication setup.
*   🏗️ **No Infrastructure**: Leverage our secure, scalable cloud.
*   📈 **Auto-Scaling**: Seamlessly scale from prototype to production.
*   🔄 **Always Updated**: Benefit from automatic updates to the latest MCP server versions.
*   💰 **Cost-Effective**: Pay only for what you use, with a free tier available.

### 💻 **Quick Integration:**

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

## 🔐 **OAuth Authentication Explained**

Some servers (like Gmail, GitHub, Slack, etc.) require OAuth for authentication. This process involves significant setup and code.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth is complex:**

*   🔧 **Complex Setup**: Requires creating OAuth apps with specific URLs and credentials.
*   📝 **Implementation Overhead**: Requires callback handling, token refresh, and secure storage.
*   🔑 **Credential Management**: Managing OAuth app secrets for different services.
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases.

Our OAuth wrapper simplifies this by handling all the complexities, letting you focus on using the MCP servers.

**Alternative**: For advanced users, you can implement OAuth yourself. Check individual server READMEs.

## 📚 **Resources & Community**

| Resource              | Link                                          | Description                            |
| --------------------- | --------------------------------------------- | -------------------------------------- |
| **📖 Documentation**   | [docs.klavis.ai](https://docs.klavis.ai)      | Complete guides and API reference      |
| **💬 Discord**        | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users        |
| **🐛 Issues**          | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features     |
| **📦 Examples**        | [examples/](examples/)                        | Working examples with AI frameworks    |
| **🔧 Server Guides**   | [mcp_servers/](mcp_servers/)                   | Individual server documentation       |

## 🤝 **Contributing**

We welcome contributions!  You can:

*   🐛 Report bugs or request features.
*   📝 Improve documentation.
*   🔧 Build new MCP servers.
*   🎨 Enhance existing servers.

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>