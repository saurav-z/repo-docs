<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unleash the Power of AI with Production-Ready MCP Servers</h1>

<p align="center"><strong>Build AI-powered applications faster with Klavis AI. Choose from self-hosted solutions, a fully managed hosted MCP service, and enterprise-grade OAuth integrations.</strong></p>

<div align="center">
  [![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
  [![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
  [![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)
</div>

## 🚀 What is Klavis AI?

Klavis AI provides production-ready Message Control Protocol (MCP) servers, streamlining integration with popular services like Gmail, GitHub, and more, allowing you to effortlessly connect AI applications to real-world data and actions. [View the original repo](https://github.com/Klavis-AI/klavis).

## ✨ Key Features

*   ✅ **Hosted Service:** Instantly deploy and manage MCP servers with our production-ready, scalable infrastructure.
*   ✅ **Self-Hosting Options:**  Run MCP servers using Docker for complete control and customization.
*   ✅ **Enterprise-Grade OAuth:** Simplify authentication with seamless integration for Google, GitHub, and other services.
*   ✅ **50+ Integrations:** Connect to a wide range of services including CRM, productivity tools, databases, and social media.
*   ✅ **Instant Deployment:** Integrate with popular AI tools like Claude Desktop, VS Code, and Cursor with zero configuration.
*   ✅ **Open Source:** Full source code available for customization and self-hosting.

## 🐳 Quick Start: Self-Hosting with Docker

Get up and running in seconds with Docker!

**1.  Get a Free API Key (for OAuth-enabled servers):** [Get Free API Key](https://www.klavis.ai/home/api-keys)

**2.  Run a GitHub MCP Server (OAuth Support):**

```bash
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
```

**3.  Or run a GitHub MCP Server (manual token):**

```bash
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
```

**Note:**  MCP servers run on port 5000 and expose the MCP protocol at the `/mcp` path (e.g., `http://localhost:5000/mcp/`).

**Example integration (Cursor):**

```json
{
  "mcpServers": {
    "github": {
      "url": "http://localhost:5000/mcp/"
    }
  }
}
```

## 🌐 Quick Start: Hosted Service (Recommended for Production)

Our hosted service provides instant access to 50+ MCP servers without the hassle of setup.

**1.  Get a Free API Key:** [Get Free API Key](https://www.klavis.ai/home/api-keys)

**2.  Install the Klavis Python or Node.js library:**

```bash
pip install klavis  # Python
# or
npm install klavis  # Node.js
```

**3.  Create and use an MCP server:**

```python
from klavis import Klavis

klavis = Klavis(api_key="Your-Klavis-API-Key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
print(f"Gmail MCP server URL: {server.server_url}")
```

**Example integration (Cursor):**

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

**4.  Get your personalized configuration:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select any service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration** for your tool
4.  **Paste into your AI tool's config** – Done!

## 🎯 Self-Hosting Instructions: Detailed

### 1.  🐳 Docker Images (Fastest Way to Start)

Perfect for quickly testing or integrating with AI tools.

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
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2.  🏗️ Build from Source

Clone and run any MCP server locally:

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

Each server directory contains detailed setup instructions.

## 🛠️ Available MCP Servers & Integrations

| Service        | Docker Image                                        | OAuth Required | Description                                      |
| -------------- | --------------------------------------------------- | -------------- | ------------------------------------------------ |
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`               | ✅             | Repository management, issues, PRs               |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server:latest`         | ✅             | Email reading, sending, management                |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                             |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`              | ❌             | Video information, search                        |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server:latest`         | ✅             | Channel management, messaging                     |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server:latest`        | ✅             | Database and page operations                       |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`    | ✅             | CRM data management                              |
| **Postgres**   | `ghcr.io/klavis-ai/postgres-mcp-server`             | ❌             | Database operations                                |
| ...            | ...                                                 | ...            | ...                                               |

And many more!

[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples: Integrate with AI Frameworks

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

### OpenAI Function Calling Example

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

## 🌐 Hosted MCP Service - No Setup Required

**Ideal for both individuals and businesses seeking instant access without infrastructure complexities.**

### ✨ **Benefits of Our Hosted Service:**

*   **🚀 Rapid Setup**: Get any MCP server running in just 30 seconds.
*   **🔐 OAuth Simplified**:  No complicated authentication setup required; we handle it.
*   **🏗️ Infrastructure-Free**: Everything operates on our secure, scalable cloud.
*   **📈 Automated Scaling**: Seamlessly scale from prototype to production.
*   **🔄 Always Up-to-Date**: Benefit from the latest MCP server versions automatically.
*   **💰 Cost-Effective**: Pay only for what you use, with a free tier available.

### 💻 **Quick Integration:**

```python
from klavis import Klavis

# Use your API key to get started
klavis = Klavis(api_key="Your-Klavis-API-Key")

# Create any MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

# Server is immediately available
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication Explained

Some servers (Google, GitHub, Slack, etc.) require OAuth for secure authentication. Implementing OAuth involves significant complexity.

```bash
# Run with OAuth support (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Challenges Addressed by Klavis AI's OAuth Integration:**

*   🔧 **Complex Setup**: We handle the creation of OAuth apps with specific redirect URLs, scopes, and credentials for each service.
*   📝 **Implementation Overhead**:  Our service manages the OAuth 2.0 flow, including callback handling, token refresh, and secure storage.
*   🔑 **Credential Management**: We take care of managing multiple OAuth app secrets across different services.
*   🔄 **Token Lifecycle**: We handle token expiration, refresh, and error cases automatically.

Our OAuth wrapper simplifies the complexities of OAuth, allowing you to focus on using the MCP servers directly.

**Alternative (Advanced Users):**  You can implement OAuth yourself by creating apps with each service provider. Refer to individual server READMEs for technical details.

## 📚 Resources & Community

| Resource             | Link                                                    | Description                                            |
| -------------------- | ------------------------------------------------------- | ------------------------------------------------------ |
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)                | Comprehensive guides and API reference                  |
| **💬 Discord**       | [Join Community](https://discord.gg/p7TuTEcssn)          | Get support and connect with other Klavis users        |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features                        |
| **📦 Examples**      | [examples/](examples/)                                  | Working examples with popular AI frameworks            |
| **🔧 Server Guides**  | [mcp_servers/](mcp_servers/)                           | Detailed documentation for individual MCP servers       |

## 🤝 Contributing

We welcome contributions!  Whether you want to:

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

See our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications Today!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  Includes relevant keywords in headings and descriptions (e.g., "MCP servers," "AI applications," "OAuth," "self-hosting," specific service names).
*   **Clear Hook:** Starts with a strong, concise sentence explaining what Klavis AI is and its main benefit.
*   **Structure:**  Uses clear, descriptive headings and subheadings.
*   **Bulleted Key Features:** Highlights the main advantages in an easy-to-scan format.
*   **Concise Language:** Removes unnecessary words and phrases.
*   **Actionable Calls to Action:**  Uses strong calls to action like "Get Free API Key," "View All Servers," and "Join Community."
*   **Formatting:** Uses markdown effectively for readability (bolding, lists, code blocks, etc.).
*   **Context & Explanation:**  Provides more context around OAuth and the benefits of the Klavis service.  Explains *why* OAuth is complex.
*   **Complete & Focused:**  Provides a comprehensive overview, enabling users to understand Klavis AI and its offerings quickly.
*   **Maintained the original structure & links:** Ensured all the existing links and original structure were maintained, including the image.
*   **Removed redundancy**: Condensed similar sections for better readability.
*   **Added Alt text to the image:** To improve accessibility.