<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Seamless AI Integration</h1>

<p align="center"><strong>Build powerful AI applications effortlessly with self-hosted and hosted MCP solutions. </strong></p>

<div align="center">
  [![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
  [![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
  [![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)
</div>

## Key Features

*   ✅ **Self-Hosted & Hosted Options:** Flexibility to run MCP servers on your infrastructure or leverage our managed service.
*   ✅ **50+ Integrations:** Connect to popular services like Gmail, GitHub, Slack, Salesforce, and more.
*   ✅ **Enterprise-Grade Features:** Secure, scalable, and compliant infrastructure for production use.
*   ✅ **Simplified OAuth:** Seamless authentication for services requiring OAuth.
*   ✅ **Instant Deployment:** Get up and running quickly with minimal configuration.
*   ✅ **Open Source**: Customize and self-host with full source code availability.

## Get Started - Run an MCP Server in Minutes

Klavis AI empowers you to effortlessly integrate your AI applications with a wide array of services. Choose your preferred deployment method:

### 🐳 **1. Self-Hosting with Docker (Fast & Flexible)**

Quickly deploy any MCP server using Docker.  This is ideal for testing, customization, and integration with your own AI tools.

**1.1.  Get Your API Key:**

*   [Get Free API Key →](https://www.klavis.ai/home/api-keys) (Required for some servers, like Gmail)

**1.2. Run GitHub MCP Server with OAuth Support**

```bash
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
```

**1.3. Run GitHub MCP Server Without OAuth**

```bash
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
```

**Note:** The MCP server runs on port 5000, exposing the MCP protocol at the `/mcp` path.

*   Example usage in Cursor:

```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:5000/mcp/"
    }
  }
}
```

### 🌐 **2. Hosted Service (Production-Ready & Simple)**

For production environments, our hosted service provides a managed infrastructure with minimal setup.  No infrastructure worries - focus on building!

**2.1. Get Your API Key:**

*   [Get Free API Key →](https://www.klavis.ai/home/api-keys)

**2.2. Install the Klavis Client**

```bash
pip install klavis
# or
npm install klavis
```

**2.3. Use in your application**

```python
from klavis import Klavis

klavis = Klavis(api_key="Your-Klavis-API-Key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

*   Example usage in Cursor:

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

**Instantly Configure Your AI Tool:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.).
3.  **Copy the generated configuration** for your tool.
4.  **Paste into your AI tool's configuration** - you're done!

## 🎯 Self Hosting Instructions (Detailed)

### 1. 🐳 Docker Images (Recommended)

This is the easiest way to get started. Docker images are available for all supported servers.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server builld by selected commit ID

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source (Advanced)

For customization and deeper control, build the MCP servers from source.

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

Detailed setup instructions for each individual server are provided in their respective `README.md` files within the `mcp_servers` directory.

## 🛠️ Available MCP Servers (Examples)

| Service        | Docker Image                                     | OAuth Required | Description                       |
| -------------- | ------------------------------------------------ | -------------- | --------------------------------- |
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`            | ✅             | Repository management, issues, PRs |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server:latest`       | ✅             | Email reading, sending, management |
| **Google Sheets**| `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations          |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`           | ❌             | Video information, search        |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server:latest`       | ✅             | Channel management, messaging    |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server:latest`      | ✅             | Database and page operations     |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`  | ✅             | CRM data management              |
| **Postgres**   | `ghcr.io/klavis-ai/postgres-mcp-server`          | ❌             | Database operations              |
| ...            | ...                                              | ...            | ...                             |

And many more!
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

### With AI Frameworks (OpenAI Function Calling Example)

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

## 🌐 Hosted MCP Service - Zero Setup

Leverage our managed infrastructure for instant access to MCP servers.

### ✨ **Why Choose Our Hosted Service:**

*   🚀 **Instant Setup**: Get any MCP server running in seconds.
*   🔐 **OAuth Simplified**: We handle the complex authentication.
*   🏗️ **No Infrastructure Management**: Run everything on our secure cloud.
*   📈 **Scalability**: Seamlessly scale from prototypes to production.
*   🔄 **Automatic Updates**: Benefit from the latest features and security patches.
*   💰 **Cost-Effective**: Pay only for what you use. Free tier available.

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

## 🔐 OAuth Authentication (Explained)

Some services (Gmail, GitHub, Slack, etc.) require OAuth for secure access. Klavis simplifies this complex process.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth Needs a Wrapper?**

*   🔧 **Complex Setup**: Each service requires creating OAuth apps, managing redirect URLs, and scopes.
*   📝 **Implementation Overhead**: OAuth 2.0 flows require callback handling, token refreshing, and secure storage.
*   🔑 **Credential Management**: Keeping track of multiple OAuth app secrets.
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases.

Our OAuth wrapper handles all the complexities so you can focus on utilizing the MCP servers directly.

## 📚 Resources & Community

| Resource                | Link                                         | Description                                     |
| ----------------------- | -------------------------------------------- | ----------------------------------------------- |
| **📖 Documentation**     | [docs.klavis.ai](https://docs.klavis.ai)     | Complete guides and API reference               |
| **💬 Discord**          | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with other users           |
| **🐛 Issues**            | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features               |
| **📦 Examples**          | [examples/](examples/)                       | Working examples with popular AI frameworks      |
| **🔧 Server Guides**     | [mcp_servers/](mcp_servers/)                  | Individual server documentation                 |
| **📦 Docker Images**     | [packages](https://github.com/orgs/Klavis-AI/packages)                 | Docker images              |

## 🤝 Contributing

We welcome contributions!

*   🐛 Report bugs or request features.
*   📝 Improve documentation.
*   🔧 Build new MCP servers.
*   🎨 Enhance existing servers.

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI! </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repo</a>
  </p>
</div>