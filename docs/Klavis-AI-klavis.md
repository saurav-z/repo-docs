<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unleash the Power of AI with Production-Ready MCP Servers</h1>

<p align="center">Quickly integrate AI with your favorite tools using self-hosted or hosted MCP (Model Control Protocol) servers. <a href="https://github.com/Klavis-AI/klavis">See the GitHub repository.</a></p>

<div align="center">
    
[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   ✅ **Hosted MCP Service:**  Production-ready infrastructure with 99.9% uptime, instant access.
*   🐳 **Self-Hosted Options:** Run MCP servers locally using Docker or build from source.
*   🔐 **Enterprise OAuth:** Seamless authentication for Google, GitHub, Slack, and more.
*   🛠️ **Extensive Integrations:** 50+ services, including CRM, productivity tools, and social media.
*   🚀 **Rapid Deployment:**  Zero-config setup for popular AI tools like Claude Desktop, VS Code, and Cursor.
*   📖 **Open Source:**  Fully open-source with customizable source code.

## Quick Start - Get an MCP Server Running in Minutes

Klavis AI provides flexible options to integrate AI with various tools and services, including self-hosted and hosted solutions.

### 🐳 Self-Hosting with Docker (Recommended for Development)

1.  **Pull a Docker image:**

    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    ```

2.  **Run the server:**

    ```bash
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
    ```
    *   Replace `$KLAVIS_API_KEY` with your API key (get one for free at [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)).

3.  **Configure your AI tool (e.g., Cursor):**

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

Access 50+ MCP servers instantly with a managed infrastructure – no setup required. Ideal for quick integrations and scalable solutions.

1.  **Install the Klavis Python package:**

    ```bash
    pip install klavis
    # or
    npm install klavis
    ```

2.  **Use the Klavis API:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")

    print(f"Gmail server URL: {server.server_url}")
    ```

    *   Get your API key at [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)

3.  **Configure your AI tool (e.g., Claude Desktop):**

    *   Visit the [MCP Servers page](https://www.klavis.ai/home/mcp-servers).
    *   Select a service (Gmail, GitHub, Slack, etc.).
    *   Copy the generated configuration.
    *   Paste the configuration into your AI tool.

## Enterprise-Grade MCP Infrastructure

Klavis AI offers a robust infrastructure designed for enterprise needs:

*   **🌐 Hosted Service:** Managed infrastructure with a 99.9% uptime SLA.
*   **🔐 Enterprise OAuth:** Simplified authentication across various services (Google, GitHub, etc.).
*   **🛠️ 50+ Integrations:** Access to CRM, productivity tools, databases, and social media platforms.
*   🚀 **Instant Deployment:** Quick setup for tools like Claude Desktop, VS Code, and Cursor.
*   🏢 **Enterprise Readiness:** Compliant with SOC2 and GDPR, plus dedicated support.

## Self-Hosting Instructions (Detailed)

### 1. 🐳 Docker Images

The easiest way to get started.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support.
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server built by selected commit ID.

[**🔍 Browse Docker Images:**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key:**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

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

Each server includes detailed setup instructions in its individual README.

## 🛠️ Available MCP Servers

| Service         | Docker Image                                      | OAuth Required | Description                          |
|-----------------|---------------------------------------------------|----------------|--------------------------------------|
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`             | ✅              | Repository management, issues, PRs   |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`       | ✅              | Email reading, sending, management  |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅              | Spreadsheet operations               |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`            | ❌              | Video information, search           |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`       | ✅              | Channel management, messaging       |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`      | ✅              | Database and page operations         |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`  | ✅              | CRM data management                  |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`           | ❌              | Database operations                  |
| ...             | ...                                               | ...            | ...                                  |

[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

Integrate Klavis AI with your existing AI tools:

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

## 🌐 Hosted MCP Service - Zero Setup Required

Benefit from a fully managed solution, ideal for those seeking speed and ease of use.

### ✨ Why Choose Our Hosted Service:

*   🚀 **Instant Setup:** Get any MCP server running in 30 seconds.
*   🔐 **OAuth Handled:** Simplified authentication setup.
*   🏗️ **No Infrastructure:** Managed on our secure and scalable cloud.
*   📈 **Auto-Scaling:** Smooth transition from prototyping to production.
*   🔄 **Always Updated:** Automatic updates to the latest MCP server versions.
*   💰 **Cost-Effective:** Pay only for what you use, with a free tier available.

### 💻 Quick Integration:

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

Klavis AI simplifies the complexities of OAuth authentication for many services.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth Needs Simplified:**

*   🔧 **Complex Setup:** Each service requires creating OAuth apps.
*   📝 **Implementation Overhead:** Requires handling callbacks, token refresh, and secure storage.
*   🔑 **Credential Management:** Managing app secrets across multiple services.
*   🔄 **Token Lifecycle:** Managing token expiration, refresh, and error cases.

Our OAuth wrapper handles these challenges, allowing you to focus on the functionality provided by MCP servers.

## 📚 Resources & Community

| Resource           | Link                                                              | Description                                  |
|--------------------|-------------------------------------------------------------------|----------------------------------------------|
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)                         | Complete guides and API reference             |
| **💬 Discord**      | [Join Community](https://discord.gg/p7TuTEcssn)                  | Get help and connect with other users          |
| **🐛 Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)       | Report bugs and request new features          |
| **📦 Examples**      | [examples/](examples/)                                           | Working examples for popular AI frameworks    |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                                     | Individual server documentation and setup     |

## 🤝 Contributing

We welcome contributions!

*   🐛 Report bugs and suggest improvements.
*   📝 Improve existing documentation.
*   🔧 Build new MCP server integrations.
*   🎨 Improve the user experience of current servers.

Check out the [Contributing Guide](CONTRIBUTING.md) for more details.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications Today!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO-Friendly Title:**  Includes keywords like "MCP Servers," "AI," and "Production-Ready."
*   **Concise Hook:** Immediately grabs attention and highlights the core value.
*   **Clear Sections & Headings:** Organized for readability and easy navigation.
*   **Bulleted Key Features:**  Quickly summarizes the benefits.
*   **Clear Instructions:**  Step-by-step guides for both Docker and hosted service, with clear code examples.
*   **API Key Emphasis:**  Highlights the need to get an API key early in the instructions, linking directly to the relevant page.
*   **Hosted Service Benefits:**  Focuses on the advantages of the hosted service.
*   **OAuth Explanation:**  Provides context and clarifies why the hosted service simplifies OAuth.
*   **Resource Links:**  Includes links to documentation, Discord, and examples.
*   **Contributing Section:**  Encourages community participation.
*   **Call to Action:**  Repeated, clear call-to-actions with links at the end.
*   **Alt Text:** Added `alt` text to the logo image for accessibility and SEO.
*   **Improved Tone:** More engaging and benefit-driven language.
*   **Updated examples:** Added example usage for TypeScript

This revised README is much more informative, user-friendly, and search-engine optimized.  It clearly explains what Klavis AI offers, guides users through getting started, and encourages them to explore further.