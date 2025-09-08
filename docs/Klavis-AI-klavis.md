<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Integration</h1>

<p align="center"><b>Seamlessly integrate your AI applications with 50+ services using Klavis AI's self-hosted and hosted MCP servers.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   <b>🚀 Instant Deployment:</b> Deploy MCP servers in seconds with Docker or our hosted service.
*   <b>🌐 Hosted Service:</b> A fully managed infrastructure with 99.9% uptime, simplifying integration.
*   <b>🔐 Enterprise-Grade OAuth:</b> Integrate services like Google, GitHub, and Slack effortlessly.
*   <b>🛠️ Extensive Integrations:</b> Connect to 50+ services, including CRMs, productivity tools, and more.
*   <b>🏢 Enterprise Ready:</b> Benefit from SOC2 compliance and GDPR readiness.
*   <b>💻 Open Source:</b> Enjoy full source code access for customization and self-hosting.

## Getting Started

### 🐳 Self-Hosted MCP Servers (Docker)

Deploy MCP servers directly using Docker for maximum flexibility and control.

1.  **Get your API Key (if needed):** [Get Free API Key →](https://www.klavis.ai/home/api-keys)

2.  **Run a Server:**

    ```bash
    # Run GitHub MCP Server (with OAuth)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
      ghcr.io/klavis-ai/github-mcp-server:latest

    # Or run GitHub MCP Server (manually add token)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
      ghcr.io/klavis-ai/github-mcp-server:latest
    ```
    **Note:** The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

    **Example in Cursor**
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

Get instant access to 50+ MCP servers with our managed infrastructure – no setup required.

1.  **Get your API Key:** [Get Free API Key →](https://www.klavis.ai/home/api-keys)

2.  **Install the Klavis Python or JavaScript library:**

    ```bash
    pip install klavis  # Python
    # or
    npm install klavis  # JavaScript
    ```

3.  **Use in your code:**

    **Python:**
    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

    **JavaScript:**
    ```javascript
    import { KlavisClient } from 'klavis';

    const klavis = new KlavisClient({ apiKey: 'your-key' });
    const server = await klavis.mcpServer.createServerInstance({
        serverName: "Gmail",
        userId: "user123"
    });
    ```

4.  **Configure in your AI tool:**

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

    **Get your personalized configuration instantly:**

    1.  [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)
    2.  Select the service you need (Gmail, GitHub, Slack, etc.).
    3.  Copy the generated configuration for your AI tool.
    4.  Paste it into your AI tool's configuration (e.g., Claude Desktop).

## 🎯 Self-Hosting Instructions (Detailed)

### 1. 🐳 Docker Images

The quickest way to deploy and test MCP servers.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server built by selected commit ID

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

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

Clone the repository and run any MCP server locally (with or without Docker).

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

Each server directory contains setup instructions in its README.

## 🛠️ Available MCP Servers (Partial List)

| Service        | Docker Image                              | OAuth Required | Description                     |
| -------------- | ----------------------------------------- | -------------- | ------------------------------- |
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`     | ✅             | Repository management, issues |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server:latest` | ✅             | Email reading and sending      |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅ | Spreadsheet operations      |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`    | ❌             | Video information, search    |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server:latest` | ✅             | Channel management, messaging  |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server:latest` | ✅             | Database and page operations |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅ | CRM data management |
| **Postgres** | `ghcr.io/klavis-ai/postgres-mcp-server` | ❌ | Database operations |
| ...            | ...                                       | ...            | ...                            |

And more!
[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

### With AI Frameworks

**Python (OpenAI Function Calling)**

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

**Ideal for individuals and businesses to quickly integrate AI without infrastructure:**

### ✨ **Why Choose Our Hosted Service:**

*   **🚀 Instant Setup:** Start using any MCP server in seconds.
*   **🔐 OAuth Simplified:** Authentication is handled automatically.
*   **🏗️ No Infrastructure to Manage:** Everything runs on our scalable cloud.
*   **📈 Auto-Scaling:** Seamlessly scale from prototype to production.
*   **🔄 Always Up-to-Date:** Benefit from the latest MCP server versions.
*   **💰 Cost-Effective:** Pay only for what you use, with a free tier available.

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

## 🔐 OAuth Authentication Explained

Klavis simplifies the complex OAuth implementation process for services like Google, GitHub, and Slack.

**Without Klavis's OAuth Support:**

*   🔧 **Complex Setup:**  Each service requires its own OAuth app configuration.
*   📝 **Implementation Overhead:**  Managing token refresh, and secure storage.
*   🔑 **Credential Management:** Securely storing OAuth app secrets.
*   🔄 **Token Lifecycle:**  Handling token expiration, refresh, and error cases.

Klavis's OAuth wrapper handles these complexities, allowing you to focus on using the MCP servers.

**Alternative (Advanced):** You can implement OAuth yourself (details available in server READMEs).

## 📚 Resources & Community

| Resource           | Link                                        | Description                                      |
| ------------------ | ------------------------------------------- | ------------------------------------------------ |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)    | Comprehensive guides and API reference           |
| **💬 Discord**       | [Join Community](https://discord.gg/p7TuTEcssn) | Get help, connect, and share with the community |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs, and request features                 |
| **📦 Examples**      | [examples/](examples/)                      | Working examples with popular AI frameworks        |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                | Detailed individual server documentation          |

## 🤝 Contributing

We welcome contributions!  Help us improve Klavis by:

*   🐛 Reporting bugs and requesting features
*   📝 Improving documentation
*   🔧 Building new MCP servers
*   🎨 Enhancing existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
  <p>
      <a href="https://github.com/Klavis-AI/klavis">View on GitHub</a>
  </p>
</div>