<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Integration</h1>

<p align="center"><strong>Seamlessly integrate your AI applications with 50+ services using Klavis AI's managed or self-hosted MCP servers.</strong></p>

<div align="center">
  <a href="https://docs.klavis.ai"><img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation"></a>
  <a href="https://www.klavis.ai"><img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website"></a>
  <a href="https://discord.gg/p7TuTEcssn"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/orgs/klavis-ai/packages"><img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images"></a>
</div>

## Key Features

*   ✅ **50+ Integrations:** Connect to popular services like Gmail, GitHub, Slack, Salesforce, and more.
*   🚀 **Instant Deployment:** Get up and running in seconds with our hosted service.
*   🔐 **Enterprise OAuth:** Simplify authentication with built-in OAuth support for major platforms.
*   🌐 **Hosted & Self-Hosted Options:** Choose the deployment model that fits your needs.
*   🛠️ **Easy Integration:** Compatible with AI frameworks like OpenAI, Cursor, and more.
*   🏢 **Enterprise-Ready:** Benefit from SOC2 compliance, GDPR readiness, and dedicated support.

## 🚀 Quick Start: Integrate with Any MCP Server in Minutes

### 🌐 Hosted Service (Recommended for Production)

Maximize speed and minimize setup with our managed infrastructure. Ideal for production environments.

**Benefits:**

*   Zero setup required
*   99.9% uptime SLA
*   Automatic updates
*   Scalable infrastructure

**Get Started:**

1.  **🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)**

```bash
pip install klavis
# or
npm install klavis
```

```python
from klavis import Klavis

klavis = Klavis(api_key="your-free-key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

### 🐳 Self-Hosting with Docker

For self-managed deployments, use Docker containers for easy setup and customization.

```bash
# Run Gmail MCP Server with OAuth Support through Klavis AI
docker run -p 5000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest
```

### 🖥️ Cursor Configuration

Integrate directly with Cursor using our hosted service URLs:

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

**💡 Find Your Configuration:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration**
4.  **Paste into your tool's config**

## 🎯 Available MCP Servers

| Service          | Docker Image                                      | OAuth Required | Description                                    |
| ---------------- | ------------------------------------------------- | -------------- | ---------------------------------------------- |
| **GitHub**       | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs             |
| **Gmail**        | `ghcr.io/klavis-ai/gmail-mcp-server:latest`      | ✅             | Email reading, sending, management           |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                        |
| **YouTube**      | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌             | Video information, search                      |
| **Slack**        | `ghcr.io/klavis-ai/slack-mcp-server:latest`      | ✅             | Channel management, messaging                 |
| **Notion**       | `ghcr.io/klavis-ai/notion-mcp-server:latest`     | ✅             | Database and page operations                   |
| **Salesforce**   | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                            |
| **Postgres**     | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌             | Database operations                            |
| **...**          | ...                                               | ...            | ...                                            |

[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 🛠️ Self-Hosting Instructions

### 1. 🐳 Docker Images (Simplest Way)

Rapidly deploy and test MCP servers using Docker containers.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth support

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Customize and run MCP servers locally by building from source.

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

> Each server directory contains detailed setup instructions in its individual README file.

### 📚 Example Usage

**Python**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123",
    platform_name="MyApp"
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

### 💡 Integration with AI Frameworks

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

## 🔐 OAuth Authentication

Klavis AI simplifies OAuth setup for services like Google, GitHub, and Slack.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Benefits of Klavis's OAuth Wrapper:**

*   Handles complex setup (OAuth apps, redirect URLs, etc.)
*   Manages token refresh and lifecycle
*   Simplifies credential management

## 📚 Resources & Community

| Resource           | Link                                       | Description                          |
| ------------------ | ------------------------------------------ | ------------------------------------ |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)    | Complete guides and API reference    |
| **💬 Discord**      | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users     |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features   |
| **📦 Examples**      | [examples/](examples/)                   | Working examples with AI frameworks |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)            | Individual server documentation    |

## 🤝 Contributing

We welcome contributions!  See our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repo</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO-Optimized Title and Description:** The title includes the primary keywords ("Klavis AI", "MCP Servers", "AI Integration") and the description highlights the key benefits for search engines.
*   **Clear and Concise Structure:**  The README is well-organized with clear headings, subheadings, and bullet points for readability.
*   **Feature Highlighting:** Key features are highlighted with bullet points, emphasizing the advantages of using Klavis AI.
*   **Actionable Quick Start:**  The "Quick Start" section is prominent and provides clear, concise instructions with code examples, making it easy for users to get started.
*   **Hosted Service Emphasis:** The benefits of the hosted service are clearly articulated, promoting its advantages.
*   **Self-Hosting Instructions:**  Detailed instructions on self-hosting with Docker and building from source are provided.
*   **Call to Action:** Clear calls to action throughout (e.g., "Get Free API Key," "View All Servers").
*   **Comprehensive Resource Section:**  Provides links to documentation, Discord, examples, and the GitHub repository.
*   **Contributing and License Information:** Keeps the important sections from the original, but formatted better.
*   **Link Back to Original Repo:** Added a final call to action linking directly back to the original GitHub repository in the final section.
*   **Alt Text for Images:** Included `alt` text for the images for accessibility and SEO.
*   **Removed Redundancy**: Consolidated information, streamlining the presentation.
*   **Revised the Introductory Text:**  Improved the introductory text to directly engage the user and showcase Klavis's value proposition.
*   **Simplified Code Examples:**  Made the code examples more concise and easier to understand.
*   **Clearer Section Titles:** Improved headings for better readability and SEO.
*   **OAuth Explanation:** Expanded the explanation of OAuth and the benefits of using Klavis's wrapper.

This revised README is more engaging, informative, and optimized for both users and search engines.