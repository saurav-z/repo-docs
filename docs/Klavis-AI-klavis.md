<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unlock Production-Ready MCP Servers for Your AI Applications</h1>

<p align="center">Easily integrate 50+ MCP servers with hosted solutions, self-hosting options, and enterprise-grade OAuth—all in one place!
</p>

<div align="center">
  [![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
  [![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
  [![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)
</div>

## Key Features

*   **🌐 Hosted MCP Service:** Production-ready, managed infrastructure with 99.9% uptime.
*   **🐳 Self-Hosting:** Docker images and source code for complete control.
*   **🔐 Enterprise OAuth:** Secure authentication for Google, GitHub, Slack, and more.
*   **🛠️ 50+ Integrations:** Connect to popular CRM, productivity tools, databases, and social media platforms.
*   **🚀 Instant Deployment:** Seamless setup for tools like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise Ready:** SOC2 compliant, GDPR ready, and supported by a dedicated team.

## 🚀 Quick Start: Get Started in Seconds!

**Klavis AI** provides multiple ways to get you up and running, offering a seamless experience tailored to your needs.

### 🌐 Hosted Service (Recommended)

**Get instant access to 50+ MCP servers with our managed infrastructure—no setup required!**

**Benefits:**
*   **Fast Integration:** Start working in 30 seconds
*   **Ease of Use:** No complicated setup steps
*   **Up-to-date:** Always getting the latest releases
*   **Cost-effective:** Pay only for usage

**Get Your Free API Key:** [Get Free API Key](https://www.klavis.ai/home/api-keys)

**Quick Start (Python)**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-free-key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

**Quick Start (Node.js)**

```bash
# Install Klavis
npm install klavis
```

```javascript
import { KlavisClient } from 'klavis';

const klavis = new KlavisClient({ apiKey: 'your-key' });
const server = await klavis.mcpServer.createServerInstance({
    serverName: "Gmail",
    userId: "user123"
});
```

### 🐳 Self-Hosting with Docker

**Run MCP servers with Docker for complete control over your infrastructure.**

```bash
# Run GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Run YouTube MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/youtube-mcp-server:latest
```

### 🖥️ Cursor Configuration

**Configure Cursor to use Klavis AI hosted services directly, eliminating the need for Docker setup.**

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

1.  **[Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration** for your tool
4.  **Paste into your tool's config** - done!

## ✨ Enterprise-Grade Infrastructure

Klavis AI offers robust features tailored for enterprise use:

*   **High Availability:** Hosted service with a 99.9% uptime SLA.
*   **Security:** SOC2 and GDPR compliance.
*   **Dedicated Support:** Professional support to assist with your needs.
*   **Easy Integration:** Zero-config setup.
*   **Open-Source Code:** Allows customization and self-hosting.

## 🎯 Self-Hosting Instructions

Choose the self-hosting method that best suits your needs:

### 1. 🐳 Docker Images (Easiest)

Perfect for fast testing and easy integration with your AI tools, like Claude Desktop.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth support

[**Browse All Docker Images**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source (Most Flexible)

Clone and run any MCP server locally, with or without Docker, for maximum control.

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

## 🛠️ Available MCP Servers

| Service        | Docker Image                                   | OAuth Required | Description                        |
| -------------- | ---------------------------------------------- | -------------- | ---------------------------------- |
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server:latest`      | ✅             | Email reading, sending, management |
| **Google Sheets**| `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations            |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌             | Video information, search          |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server:latest`      | ✅             | Channel management, messaging      |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server:latest`     | ✅             | Database and page operations      |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                |
| **Postgres**   | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌             | Database operations                |
| ...            | ...                                            | ...            | ...                                |

And more!

*   [**View All 50+ Servers**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)
*   [**Browse Docker Images**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

Integrate Klavis AI with your preferred AI frameworks:

**Python Example:**
```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123",
    platform_name="MyApp"
)
```

**TypeScript Example:**
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

*   [**View Complete Examples**](examples/)

## 🌐 Hosted Service - Zero Setup Required

**Enjoy instant access to all MCP servers without the hassle of infrastructure management.**

### ✨ **Why Choose Our Hosted Service?**

*   **Fast Integration:** Get started with our hosted service in 30 seconds.
*   **OAuth Simplified:** OAuth setup is handled seamlessly.
*   **Fully Managed:** Focus on building, not managing infrastructure.
*   **Scalable:** Easily scale from prototype to production.
*   **Always Updated:** Benefit from the latest server versions automatically.
*   **Cost-Effective:** Pay only for what you use, with a free tier.

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

*   [**Get Free API Key**](https://www.klavis.ai/home/api-keys)
*   [**Complete Documentation**](https://docs.klavis.ai)

## 🔐 OAuth Authentication

Simplify OAuth implementation with Klavis AI for secure authentication:

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why Klavis Simplifies OAuth:**

*   **Simplified Setup:** Manage authentication details easily.
*   **Simplified Flow:** We take care of token refreshes and expiration.
*   **Reduced Complexity:** No need to handle secrets or callbacks.

**Alternative:** Advanced users can implement OAuth themselves. Check the server's README for details.

## 📚 Resources & Community

| Resource         | Link                                     | Description                             |
| ---------------- | ---------------------------------------- | --------------------------------------- |
| **Documentation**  | [docs.klavis.ai](https://docs.klavis.ai) | Complete guides and API reference       |
| **Discord**      | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users       |
| **GitHub Issues** | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features      |
| **Examples**     | [examples/](examples/)                 | Working examples with popular AI frameworks |
| **Server Guides**  | [mcp_servers/](mcp_servers/)           | Individual server documentation           |

## 🤝 Contributing

We welcome contributions! Whether you want to:

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to learn more.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
    • <a href="https://github.com/Klavis-AI/klavis">GitHub Repo</a>
  </p>
</div>
```
Key improvements:

*   **SEO Optimization:**  Keyword-rich headings (e.g., "Klavis AI: Unlock Production-Ready MCP Servers..."), introduction uses targeted keywords, and the document is well-structured for search engine readability.  Added the phrase "Supercharge your AI Applications"
*   **Concise Hook:** The one-sentence hook is now "Easily integrate 50+ MCP servers with hosted solutions, self-hosting options, and enterprise-grade OAuth—all in one place!"
*   **Clearer Structure:**  Uses clear headings, subheadings, and bullet points for readability and scannability.
*   **Benefit-Oriented Language:** Highlights the advantages of Klavis AI's offerings, not just the features.
*   **Actionable Steps:**  Includes clear calls to action ("Get Free API Key," "Browse Docker Images," etc.).
*   **Comprehensive Coverage:** The most important sections of the original README have been kept.
*   **Improved Code Samples:**  Includes example code snippets and how to get started with Python and Node.js.
*   **Emphasis on Ease of Use:** Highlights Klavis AI's user-friendliness and speed.
*   **Enhanced Resource Links:**  Links are organized and easily accessible.
*   **Added GitHub Repo Link:**  Added a link back to the GitHub repo.
*   **Clearer Table:** The MCP servers table is more readable.
*   **Removed Redundancy:** Streamlined text to avoid unnecessary repetition.
*   **Visual Appeal:** Keeps the image, and adds a more professional feel.