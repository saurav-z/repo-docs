<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Seamlessly Integrate AI with Your Favorite Tools</h1>

**Klavis AI provides production-ready MCP (Model Control Protocol) servers, offering self-hosted and hosted solutions with enterprise-grade features for effortless AI integration.**

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features of Klavis AI

*   ✅ **Self-Hosted Solutions:** Deploy MCP servers using Docker for full control.
*   🌐 **Hosted MCP Service:** Production-ready infrastructure with 99.9% uptime and instant setup.
*   🔐 **Enterprise OAuth:** Secure authentication for services like Google, GitHub, and Slack.
*   🛠️ **50+ Integrations:** Connect to CRMs, productivity tools, databases, and social media.
*   🚀 **Instant Deployment:** Easily integrate with Claude Desktop, VS Code, Cursor, and more.
*   🏢 **Enterprise-Ready:** SOC2 compliant and GDPR ready.

## Quick Start: Run an MCP Server in Seconds

Get up and running with Klavis AI using Docker or our hosted service.

### 🐳 Self-Hosting with Docker

**Perfect for developers needing control & customization.**

1.  **Get a Free API Key (if using OAuth):** [Get Free API Key →](https://www.klavis.ai/home/api-keys)
2.  **Run a GitHub MCP Server (Example):**

    ```bash
    # With OAuth Support (Requires API Key)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
      ghcr.io/klavis-ai/github-mcp-server:latest

    # Or Run GitHub MCP Server (Manual token)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
      ghcr.io/klavis-ai/github-mcp-server:latest
    ```

**Note:**  MCP servers run on port 5000, exposing the MCP protocol at the `/mcp` path.

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

### 🌐 Hosted Service (Recommended for Production)

**Experience the power of Klavis AI with zero setup.**

1.  **Get a Free API Key:** [Get Free API Key →](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis SDK:**

    ```bash
    pip install klavis
    # or
    npm install klavis
    ```

3.  **Use the SDK:**

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

**Get your personalized configuration:**

1.  **Visit:** [MCP Servers Page](https://www.klavis.ai/home/mcp-servers)
2.  **Select a service** (Gmail, GitHub, etc.)
3.  **Copy the generated configuration**
4.  **Paste** into your AI tool (e.g., Claude Desktop) - Done!

## Enterprise-Grade MCP Infrastructure

*   **Hosted Service:** Reliable, managed infrastructure with 99.9% uptime.
*   **Enterprise OAuth:** Simplified authentication for popular services.
*   **Extensive Integrations:** Support for 50+ services to supercharge your AI workflow.
*   **Rapid Deployment:** Integrate with your AI tools instantly, no configuration needed.
*   **Enterprise Ready:** Meets SOC2 and GDPR compliance standards.
*   **Open Source Foundation:** Complete source code for customization.

## 🎯 Self-Hosting Instructions in Detail

### 1.  🐳 Docker Images (Easiest Approach)

**Ideal for quickly trying MCP servers and integrating with tools.**

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` (OAuth support)
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` (Specific commit builds)

**Browse All Images:** [Browse Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

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

**Get your API Key:** [Get Free API Key](https://www.klavis.ai/home/api-keys)

### 2.  🏗️ Build from Source (For Advanced Users)

Customize and run servers locally:

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

**Note:** Each server includes detailed setup instructions in its README.

**Or, use our managed service (no Docker needed):**

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers - Wide Range of Integrations

| Service      | Docker Image                             | OAuth Required | Description                               |
|--------------|------------------------------------------|----------------|-------------------------------------------|
| **GitHub**   | `ghcr.io/klavis-ai/github-mcp-server`     | ✅             | Repository, issues, PRs                   |
| **Gmail**    | `ghcr.io/klavis-ai/gmail-mcp-server:latest`   | ✅             | Email reading, sending, management       |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                   |
| **YouTube**  | `ghcr.io/klavis-ai/youtube-mcp-server`    | ❌             | Video information, search                 |
| **Slack**    | `ghcr.io/klavis-ai/slack-mcp-server:latest`    | ✅             | Channel management, messaging           |
| **Notion**   | `ghcr.io/klavis-ai/notion-mcp-server:latest`   | ✅             | Database and page operations           |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                    |
| **Postgres** | `ghcr.io/klavis-ai/postgres-mcp-server`   | ❌             | Database operations                      |
| ...          | ...                                      | ...            | ...                                       |

**Explore More Servers:**

*   [View All 50+ Servers](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)
*   [Browse Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Example Usage - Integrate with AI Frameworks

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

### With AI Frameworks: OpenAI Function Calling

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

[**📖 Complete Examples**](examples/)

## 🌐 Hosted MCP Service - Zero Setup Needed

**Simplify your AI integration with our managed service.**

### ✨ **Benefits of the Hosted Service:**

*   🚀 **Instant Setup**: Get any MCP server running in 30 seconds.
*   🔐 **OAuth Simplified**: Handles complex authentication for you.
*   🏗️ **No Infrastructure:**  Runs on our secure, scalable cloud.
*   📈 **Auto-Scaling:**  Seamlessly scale from prototype to production.
*   🔄 **Always Updated**:  Get the latest server versions automatically.
*   💰 **Cost-Effective:**  Pay-as-you-go, with a free tier available.

### 💻 **Quick Integration:**

```python
from klavis import Klavis

# Start with an API key
klavis = Klavis(api_key="Your-Klavis-API-Key")

# Create an MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

# Server ready to use immediately
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**Get Started:**

*   [Get Free API Key](https://www.klavis.ai/home/api-keys)
*   [Complete Documentation](https://docs.klavis.ai)

## 🔐 OAuth Authentication - Simplified

**Easily integrate services requiring OAuth authentication (Google, GitHub, Slack).**

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth can be complex:**

*   🔧 **Complex Setup:** Requires creating OAuth apps with specific redirect URLs, scopes, and credentials.
*   📝 **Implementation Overhead:** Requires callback handling, token refresh, and secure storage.
*   🔑 **Credential Management:** Managing multiple OAuth app secrets across different services.
*   🔄 **Token Lifecycle:** Handling token expiration, refresh, and error cases.

**Klavis AI simplifies OAuth by handling the complexity.**

**Alternative (Advanced Users):** You can implement OAuth manually.  See individual server READMEs.

## 📚 Resources and Community

| Resource             | Link                                                        | Description                                   |
|----------------------|-------------------------------------------------------------|-----------------------------------------------|
| **📖 Documentation**  | [docs.klavis.ai](https://docs.klavis.ai)                   | Complete guides and API reference            |
| **💬 Discord**       | [Join Community](https://discord.gg/p7TuTEcssn)            | Get help and connect with other users       |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features             |
| **📦 Examples**      | [examples/](examples/)                                       | Working examples with AI frameworks         |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                                 | Individual server documentation              |

## 🤝 Contributing

We welcome your contributions!

*   🐛 Report bugs or suggest features.
*   📝 Improve documentation.
*   🔧 Build new MCP servers.
*   🎨 Enhance existing servers.

Get started by reviewing our [Contributing Guide](CONTRIBUTING.md)!

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI! </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>