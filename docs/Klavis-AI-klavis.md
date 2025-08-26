<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unleash the Power of Production-Ready MCP Servers</h1>

<p align="center"><b>Seamlessly integrate with 50+ services using Klavis AI's hosted or self-hosted MCP servers.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   **🌐 Hosted Service:** Production-ready, managed infrastructure with 99.9% uptime SLA. Access servers instantly without setup.
*   **🐳 Self-Hosted Solutions:** Deploy and customize MCP servers using Docker or build from source.
*   **🔐 Enterprise OAuth:** Secure and simplified authentication for services like Google, GitHub, and Slack.
*   **🛠️ 50+ Integrations:** Connect to a wide range of CRM, productivity tools, databases, and social media platforms.
*   **🚀 Instant Deployment:** Quick setup for popular AI tools like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise Ready:** SOC2 compliant, GDPR ready, and includes dedicated support.
*   **📖 Open Source:** Full source code available for customization and self-hosting.

## 🚀 Getting Started: Run MCP Servers Instantly

Choose your preferred deployment method for Klavis AI MCP servers.

### 🌐 Hosted Service (Recommended)

**Ideal for rapid integration and production environments.**  Get up and running in seconds.

**1. Get Your Free API Key:**
**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)**

**2. Install the Klavis Client:**

```bash
pip install klavis
# or
npm install klavis
```

**3. Instantiate & Use:**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-free-key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

### 🐳 Self-Hosting with Docker

**For developers seeking control and customization.**

**1.  Run a pre-built Docker image:**

```bash
# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 🖥️ Cursor Configuration

**Direct integration with Cursor AI tools is simplified with the Klavis hosted service:**

**1.  Use Hosted Service URLs:**

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

**2.  Generate Configuration:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration**
4.  **Paste into your tool's config**

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images (Recommended for self-hosting)

Quickly deploy pre-built MCP servers.

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

For advanced customization and control.

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

Refer to each server's individual README for detailed setup instructions.

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers

| Service         | Docker Image                                        | OAuth Required | Description                       |
| --------------- | --------------------------------------------------- | :------------: | --------------------------------- |
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`             |      ✅       | Repository management, issues, PRs |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`        |      ✅       | Email reading, sending, management |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` |      ✅       | Spreadsheet operations              |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`              |      ❌       | Video information, search         |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`          |      ✅       | Channel management, messaging      |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`          |      ✅       | Database and page operations       |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`      |      ✅       | CRM data management               |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`             |      ❌       | Database operations               |
| ...             | ...                                                 |      ...       | ...                               |

And more!

[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

### Python

```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123",
    platform_name="MyApp"
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

**Get started with Klavis AI's Hosted Service and eliminate infrastructure complexity.**

### ✨ **Key Advantages:**

*   **🚀 Instant Setup**: Deploy any MCP server in under 30 seconds.
*   **🔐 OAuth Handled**: Simplify authentication with our managed OAuth flow.
*   **🏗️ No Infrastructure**: Operate seamlessly on our secure, scalable cloud.
*   **📈 Auto-Scaling**: Scale effortlessly from prototype to production.
*   **🔄 Always Updated**: Benefit from the latest MCP server versions.
*   **💰 Cost-Effective**: Pay-as-you-go, with a free tier available.

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

**Klavis AI simplifies complex OAuth implementations for various services.**

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Challenges of manual OAuth Implementation:**

*   🔧 **Complex Setup**: OAuth requires creating apps, redirect URLs, and credentials for each service.
*   📝 **Implementation Overhead**: Handling callbacks, token refreshing, and secure storage adds complexity.
*   🔑 **Credential Management**: Managing multiple OAuth app secrets.
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and errors.

## 📚 Resources & Community

| Resource          | Link                                        | Description                      |
| ----------------- | ------------------------------------------- | -------------------------------- |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)     | Comprehensive guides and API ref |
| **💬 Discord**      | [Join Community](https://discord.gg/p7TuTEcssn) | Get help, connect with users     |
| **🐛 Issues**       | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)    | Report bugs and request features |
| **📦 Examples**     | [examples/](examples/)                    | Code examples with AI frameworks  |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)              | Individual server documentation |

## 🤝 Contributing

We welcome contributions!  Help improve Klavis AI!

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>
```
Key improvements and optimizations:

*   **SEO-Focused Hook:**  "Seamlessly integrate with 50+ services using Klavis AI's hosted or self-hosted MCP servers."  This clearly states the core value and includes important keywords.
*   **Clear Headings:**  Uses `<h1>` and `<h2>` tags for better structure and SEO.
*   **Bulleted Key Features:**  Uses bullet points for easy readability and highlights key benefits.
*   **Concise and Direct Language:**  The text is streamlined for clarity.
*   **Stronger Call to Actions:**  Encourages users to get a free API key and explore the documentation.
*   **Keyword Optimization:**  Includes relevant keywords throughout (e.g., "MCP servers," "hosted service," "self-hosting," specific service names).
*   **Internal Linking:**  Links to other parts of the README for easier navigation.
*   **Consistent Formatting:** Improved formatting for better readability.
*   **Emphasis on Benefits:** Focuses on what the user gains.
*   **Reduced Redundancy**: Streamlined instructions.
*   **Alt text added for images**: to help with accessibility.
*   **Revised Introduction to reflect the project's benefits.**
*   **More detailed instructions** to support both the hosted and self-hosting methods.
*   **Updated the example with the use of OPENAI**