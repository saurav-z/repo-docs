<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Integration</h1>

<p align="center"><strong>Seamlessly integrate AI with your favorite tools using our self-hosted and hosted MCP server solutions.</strong></p>

<div align="center">
  <a href="https://docs.klavis.ai">
    <img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation">
  </a>
  <a href="https://www.klavis.ai">
    <img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website">
  </a>
  <a href="https://discord.gg/p7TuTEcssn">
    <img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://github.com/orgs/klavis-ai/packages">
    <img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images">
  </a>
</div>

## What is Klavis AI?

Klavis AI provides production-ready MCP (Multi-Channel Protocol) servers, enabling you to effortlessly connect your AI applications with various services like Gmail, GitHub, Slack, and more.  Choose between self-hosting or our fully managed hosted service for instant access.

**Key Features:**

*   🚀 **Instant Deployment:**  Get up and running in seconds with Docker or our hosted service.
*   🌐 **Hosted Service:** Production-ready managed infrastructure with 99.9% uptime, zero setup required.
*   🔐 **Enterprise OAuth:** Seamless authentication for Google, GitHub, Slack, and more.
*   🛠️ **50+ Integrations:**  Connect to CRMs, productivity tools, databases, social media, and more.
*   🐳 **Self-Hosting Options:**  Docker images and source code available for complete control.
*   🏢 **Enterprise Ready:**  SOC2 compliant, GDPR ready, and offers dedicated support.

## 🚀 Quick Start: Run an MCP Server in 30 Seconds

### 🐳 Self-Hosting with Docker

Easily self-host MCP servers for maximum control.

**Get started with a free API key:  [Get Free API Key →](https://www.klavis.ai/home/api-keys)**

```bash
# Run GitHub MCP Server with OAuth Support
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

**Note:** The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

**Example Usage in Cursor:**

```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:5000/mcp/"
    }
  }
}
```

### 🌐 Using the Hosted Service (Recommended for Production)

Benefit from our managed infrastructure with instant access to 50+ MCP servers – no setup required!

**Get started with a free API key:  [Get Free API Key →](https://www.klavis.ai/home/api-keys)**

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

**Example Usage in Cursor:**

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

1.  **🔗 [Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)**
2.  **Select any service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration** for your tool
4.  **Paste into Claude Desktop config** - done!

## 🎯 Self-Hosting Instructions in Detail

### 1. 🐳 Docker Images

The fastest and easiest way to get started.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server built by selected commit ID

**Browse Docker Images:**  [**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

**Get your API Key**:  [**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

For advanced customization and control, build from source.

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

Each server directory provides specific setup instructions within its `README`.

## 🛠️ Available MCP Servers:  A Growing List

| Service        | Docker Image                                    | OAuth Required | Description                                       |
| -------------- | ----------------------------------------------- | -------------- | ------------------------------------------------- |
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`          | ✅            | Repository management, issues, PRs                 |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server:latest`    | ✅            | Email reading, sending, management                |
| **Google Sheets**| `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅            | Spreadsheet operations                              |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`         | ❌            | Video information, search                         |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server:latest`    | ✅            | Channel management, messaging                     |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server:latest`   | ✅            | Database and page operations                      |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅            | CRM data management                               |
| **Postgres**   | `ghcr.io/klavis-ai/postgres-mcp-server`        | ❌            | Database operations                               |
| ...            | ...                                             | ...            | ...                                               |

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

## 🌐 Hosted MCP Service:  Effortless Integration

**Our hosted service is perfect for those seeking simplicity and speed:**

### ✨ Why Choose Our Hosted Service?

*   🚀 **Instant Setup:** Get any MCP server running in 30 seconds.
*   🔐 **OAuth Handled:**  No complex authentication setup required.
*   🏗️ **No Infrastructure:**  Everything runs on our secure, scalable cloud.
*   📈 **Auto-Scaling:**  Scales automatically from prototype to production.
*   🔄 **Always Updated:**  Benefit from the latest MCP server versions.
*   💰 **Cost-Effective:**  Pay only for what you use, with a free tier available.

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

## 🔐 OAuth Authentication

Some servers require OAuth.  Klavis AI simplifies this complex process.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why is OAuth implementation complex?**
* 🔧 **Complex Setup**: Requires creating OAuth apps with specific redirect URLs, scopes, and credentials.
* 📝 **Implementation Overhead**: OAuth 2.0 flow needs callback handling, token refresh, and secure storage.
* 🔑 **Credential Management**: Managing multiple OAuth app secrets.
* 🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases.

Our hosted and self-hosting solutions handle the complexity, letting you focus on AI integration.

## 📚 Resources & Community

| Resource           | Link                                                 | Description                                     |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)              | Complete guides and API reference               |
| **💬 Discord**      | [Join Community](https://discord.gg/p7TuTEcssn)       | Get help and connect with users               |
| **🐛 Issues**       | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features                |
| **📦 Examples**     | [examples/](examples/)                                 | Working examples with popular AI frameworks     |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                         | Individual server documentation                 |

## 🤝 Contributing

We welcome contributions!  Help us build the future of AI integration!

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>

[**Back to Top**](#klavis-ai-production-ready-mcp-servers-for-ai-integration)
```
Key improvements and SEO considerations:

*   **Compelling Hook:** Starts with a clear and concise value proposition.
*   **Keyword Optimization:** Uses relevant keywords like "MCP server," "AI integration," "self-hosted," "hosted service," and service names (Gmail, GitHub, etc.).  Repeats keywords naturally.
*   **Clear Headings and Structure:** Organizes information logically for readability and search engine indexing.
*   **Bullet Points:**  Highlights key features effectively.
*   **Call to Actions (CTAs):**  Includes numerous CTAs throughout ("Get Free API Key," "Visit our MCP Servers page," etc.).
*   **Internal Linking:** Links to relevant sections within the README.
*   **External Linking:**  Links to important resources (documentation, Discord, etc.) for SEO and user experience.
*   **Alt Text:**  Added `alt` attributes to images for accessibility and SEO.
*   **Concise Language:**  Rephrases content for clarity and impact.
*   **Updated Badges:** Kept and improved badge usage for vital links
*   **Comprehensive Coverage:** Addresses all sections of the original README.
*   **Back to Top anchor:** Adds anchor to the top of the document to improve UX
*   **More Examples:** Added more example usage.
*   **Clear distinctions:** Self-hosting vs hosted-service.