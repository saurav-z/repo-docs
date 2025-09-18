<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80">
  </picture>
</div>

<h1 align="center">Klavis AI: Unlock Powerful Integrations for AI with Production-Ready MCP Servers</h1>
<p align="center"><b>Build AI applications faster with seamless access to 50+ services, self-hosting options, and enterprise-grade features.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## 🚀 Key Features of Klavis AI

*   🌐 **Hosted Service:** Deploy MCP servers with a managed infrastructure that is production-ready with a 99.9% uptime SLA.
*   🐳 **Self-Hosted Solutions:**  Run MCP servers using Docker for ultimate control and customization.
*   🔐 **Enterprise OAuth:**  Simplify authentication for Google, GitHub, Slack, and other popular services.
*   🛠️ **50+ Integrations:** Access a wide range of integrations, including CRM, productivity tools, databases, and social media.
*   🚀 **Instant Deployment:** Integrate with your favorite tools like Claude Desktop, VS Code, and Cursor with zero configuration.
*   🏢 **Enterprise-Ready:** Benefit from SOC2 compliance, GDPR readiness, and dedicated support.
*   📦 **Open Source Foundation:** Leverage a robust open-source framework for customization and self-hosting.

## ⚡️ Get Started Quickly

Klavis AI offers flexibility whether you prefer self-hosting or a managed service.

### 🐳 Option 1: Self-Hosting with Docker

1.  **Get Your API Key:** [Get Free API Key](https://www.klavis.ai/home/api-keys) (required for some servers)
2.  **Pull the Docker Image:**

    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    ```

3.  **Run the Server (with OAuth support):**

    ```bash
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
    ```

    *   The MCP server runs on port 5000.

4.  **Or, run GitHub MCP server (manually add token)**

    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
    ghcr.io/klavis-ai/github-mcp-server:latest
    ```

### 🌐 Option 2: Using the Hosted Service (Recommended)

1.  **Install the Klavis Python Package:**

    ```bash
    pip install klavis
    ```

    **Or install the Klavis Javascript Package:**

    ```bash
    npm install klavis
    ```

2.  **Use the Klavis Client:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

    or

    ```javascript
    import { KlavisClient } from 'klavis';

    const klavis = new KlavisClient({ apiKey: 'your-key' });
    const server = await klavis.mcpServer.createServerInstance({
        serverName: "Gmail",
        userId: "user123"
    });
    ```

    *   You can view the full server configuration at [Klavis AI's MCP Servers page](https://www.klavis.ai/home/mcp-servers).

    *   Paste the generated configuration into your tool.

## ⚙️ Available MCP Servers

| Service         | Docker Image                                     | OAuth Required | Description                                          |
|-----------------|--------------------------------------------------|----------------|------------------------------------------------------|
| GitHub          | `ghcr.io/klavis-ai/github-mcp-server`              | ✅              | Repository management, issues, PRs                   |
| Gmail           | `ghcr.io/klavis-ai/gmail-mcp-server:latest`          | ✅              | Email reading, sending, management                   |
| Google Sheets   | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅              | Spreadsheet operations                               |
| YouTube         | `ghcr.io/klavis-ai/youtube-mcp-server`             | ❌              | Video information, search                            |
| Slack           | `ghcr.io/klavis-ai/slack-mcp-server:latest`          | ✅              | Channel management, messaging                        |
| Notion          | `ghcr.io/klavis-ai/notion-mcp-server:latest`         | ✅              | Database and page operations                         |
| Salesforce      | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`     | ✅              | CRM data management                                  |
| Postgres        | `ghcr.io/klavis-ai/postgres-mcp-server`            | ❌              | Database operations                                  |
| ...             | ...                                                | ...             | ...                                                  |

Find over 50+ other servers [here](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) and browse Docker Images [here](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis).

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

### Using with OpenAI Function Calling

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

See [examples/](examples/) for more information.

## 🛠️ Self Hosting Instructions

### 1. 🐳 Docker Images (Fastest Way to Start)

Perfect for trying out MCP servers or integrating with AI tools like Claude Desktop.

**Available Images:**
- `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
- `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server builld by selected commit ID

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

Use our managed infrastructure - no Docker required:

```bash
pip install klavis  # or npm install klavis
```

## 🌐 Hosted MCP Service - Zero Setup Required

**Perfect for individuals and businesses who want instant access without infrastructure complexity:**

### ✨ **Why Choose Our Hosted Service:**
- **🚀 Instant Setup**: Get any MCP server running in 30 seconds
- **🔐 OAuth Handled**: No complex authentication setup required  
- **🏗️ No Infrastructure**: Everything runs on our secure, scalable cloud
- **📈 Auto-Scaling**: From prototype to production seamlessly
- **🔄 Always Updated**: Latest MCP server versions automatically
- **💰 Cost-Effective**: Pay only for what you use, free tier available

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

## 🔐 OAuth Authentication (For OAuth-Enabled Servers)

Some servers require OAuth authentication (Google, GitHub, Slack, etc.). OAuth implementation requires significant setup and code complexity:

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth needs additional implementation?**
- 🔧 **Complex Setup**: Each service requires creating OAuth apps with specific redirect URLs, scopes, and credentials
- 📝 **Implementation Overhead**: OAuth 2.0 flow requires callback handling, token refresh, and secure storage
- 🔑 **Credential Management**: Managing multiple OAuth app secrets across different services
- 🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases

Our OAuth wrapper simplifies this by handling all the complex OAuth implementation details, so you can focus on using the MCP servers directly.

**Alternative**: For advanced users, you can implement OAuth yourself by creating apps with each service provider. Check individual server READMEs for technical details.

## 📚 Resources & Community

| Resource           | Link                                         | Description                                     |
|--------------------|----------------------------------------------|-------------------------------------------------|
| Documentation      | [docs.klavis.ai](https://docs.klavis.ai)        | Complete guides and API reference                 |
| Discord            | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users                 |
| GitHub Issues      | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)    | Report bugs and request features               |
| Examples           | [examples/](examples/)                       | Working examples with popular AI frameworks        |
| Server Guides      | [mcp_servers/](mcp_servers/)                   | Individual server documentation                   |

## 🤝 Contributing

We welcome contributions! If you want to:

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

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