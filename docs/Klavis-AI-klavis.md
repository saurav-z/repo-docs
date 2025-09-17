<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Seamless AI Integration</h1>

<p align="center"><b>Unlock the power of AI with Klavis: self-hosted & hosted MCP solutions, plus enterprise-grade OAuth.</b></p>

<div align="center">
  <a href="https://docs.klavis.ai"><img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation"></a>
  <a href="https://www.klavis.ai"><img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website"></a>
  <a href="https://discord.gg/p7TuTEcssn"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/orgs/klavis-ai/packages"><img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images"></a>
</div>

## Key Features:

*   ✅ **Hosted MCP Service:** Get instant access to 50+ pre-built MCP servers without any setup.
*   🐳 **Self-Hosted Solutions:** Deploy servers quickly using Docker or build from source for full control.
*   🔐 **Enterprise-Grade OAuth:** Simplified authentication for popular services like Gmail, GitHub, and Slack.
*   🚀 **Rapid Integration:** Seamlessly connect to AI frameworks like OpenAI, Claude, and more.
*   🛠️ **Extensive Integrations:** Connect to CRMs, productivity tools, databases, and social media platforms.
*   🌐 **Scalable & Reliable:** Production-ready infrastructure with a 99.9% uptime SLA for hosted solutions.

## Quick Start: Run Any MCP Server in 30 Seconds

### 🐳 **Self-Hosting with Docker**

Get started quickly with pre-built Docker images.

**Get Started:**
1.  **[Get Your Free API Key](https://www.klavis.ai/home/api-keys)** (for OAuth-enabled servers)
2.  **Pull the Docker Image:**

    ```bash
    # Example: GitHub MCP Server (with OAuth)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest

    # Example: GitHub MCP Server (manual token)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
    ```

*   **Note:** The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

### 🌐 **Hosted Service (Recommended for Production)**

Access a managed, production-ready infrastructure with zero setup.

**Get Your API Key:**
1.  **[Get Free API Key](https://www.klavis.ai/home/api-keys)**

2.  **Install the Klavis Client:**

    ```bash
    pip install klavis  # or npm install klavis
    ```

3.  **Example Usage:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

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

4.  **Instant Configuration:**
    1.  **[Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)**
    2.  Select your desired service (Gmail, GitHub, Slack, etc.)
    3.  Copy the generated configuration.
    4.  Paste it into your AI tool (e.g., Claude Desktop) and you're done!

## Enterprise-Grade MCP Infrastructure

*   **Hosted Service:** Managed infrastructure with 99.9% uptime SLA
*   **Enterprise OAuth:** Simplified authentication for Google, GitHub, and more.
*   **50+ Integrations:** Connect to CRMs, productivity tools, databases, and social media platforms.
*   **Instant Deployment:** Zero-config setup for tools like Claude Desktop, VS Code, and Cursor.
*   **Enterprise Ready:** SOC2 compliant, GDPR ready, and dedicated support.
*   **Open Source:** Customize and self-host with full source code.

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images (Fastest Way)

Perfect for a quick start and integration with AI tools.

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

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Clone and run any MCP server locally (with or without Docker).

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

### 💡 Use Our Hosted Infrastructure (No Docker Required)

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers

| Service          | Docker Image                                     | OAuth Required | Description                                 |
| ---------------- | ------------------------------------------------ | -------------- | ------------------------------------------- |
| **GitHub**       | `ghcr.io/klavis-ai/github-mcp-server`             | ✅             | Repository management, issues, PRs        |
| **Gmail**        | `ghcr.io/klavis-ai/gmail-mcp-server:latest`        | ✅             | Email reading, sending, management          |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                      |
| **YouTube**      | `ghcr.io/klavis-ai/youtube-mcp-server`            | ❌             | Video information, search                   |
| **Slack**        | `ghcr.io/klavis-ai/slack-mcp-server:latest`        | ✅             | Channel management, messaging               |
| **Notion**       | `ghcr.io/klavis-ai/notion-mcp-server:latest`       | ✅             | Database and page operations                |
| **Salesforce**   | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`   | ✅             | CRM data management                         |
| **Postgres**     | `ghcr.io/klavis-ai/postgres-mcp-server`           | ❌             | Database operations                         |
| ...              | ...                                              | ...            | ...                                         |

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

## 🌐 Hosted MCP Service - Zero Setup Required

**Perfect for individuals and businesses seeking immediate access without the complexities of infrastructure management.**

### ✨ **Why Choose Our Hosted Service:**

*   🚀 **Instant Setup:** Get any MCP server up and running in seconds.
*   🔐 **OAuth Handled:** No complex authentication setup is required.
*   🏗️ **No Infrastructure:** Everything is hosted on our secure, scalable cloud.
*   📈 **Auto-Scaling:** Seamless scalability from prototype to production.
*   🔄 **Always Updated:** Automatically receive the latest MCP server versions.
*   💰 **Cost-Effective:** Pay only for the resources you consume, with a free tier available.

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

Klavis AI simplifies complex OAuth authentication for services like Google, GitHub, and Slack.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Benefits of Our OAuth Implementation:**

*   🔧 **Simplified Setup:** No need to create OAuth apps or manage redirect URLs.
*   📝 **Implementation Abstraction:** Token refresh, storage, and error handling are managed for you.
*   🔑 **Credential Management:** Say goodbye to managing multiple OAuth app secrets.
*   🔄 **Automated Token Handling:** Tokens are automatically managed and refreshed.

**Alternative:** For advanced users, you can implement OAuth yourself, but Klavis simplifies the process significantly. Check the server-specific READMEs for implementation details.

## 📚 Resources & Community

| Resource                | Link                                                      | Description                                         |
| ----------------------- | --------------------------------------------------------- | --------------------------------------------------- |
| **📖 Documentation**       | [docs.klavis.ai](https://docs.klavis.ai)                  | Comprehensive guides and API reference              |
| **💬 Discord**           | [Join Community](https://discord.gg/p7TuTEcssn)           | Get help and connect with other users                |
| **🐛 Issues**            | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request new features                 |
| **📦 Examples**          | [examples/](examples/)                                    | Working examples using popular AI frameworks         |
| **🔧 Server Guides**     | [mcp_servers/](mcp_servers/)                              | Documentation for individual server implementations |
| **🐳  Docker Images**     | [Packages](https://github.com/orgs/Klavis-AI/packages)  | Get the latest docker images                                        |

## 🤝 Contributing

We welcome contributions! If you're interested in:

*   🐛 Reporting bugs or requesting features
*   📝 Improving our documentation
*   🔧 Building new MCP servers
*   🎨 Enhancing existing servers

Check our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications Today! </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
    <br>
    <a href="https://github.com/klavis-ai/klavis">View on GitHub</a>
  </p>
</div>
```
Key improvements and SEO considerations:

*   **Clear Headline with Keywords:**  Uses "Klavis AI" and target keywords (MCP, AI Integration, Production-Ready, Self-Hosted, Hosted) in the main heading and subsequent section headings.
*   **One-Sentence Hook:** The initial paragraph immediately grabs the reader's attention.
*   **Structured with Headings & Subheadings:**  Organization is greatly improved, making it easy to scan and understand.
*   **Bulleted Key Features:** Uses bullet points for clear presentation of benefits.
*   **Keyword Optimization:** Keywords are naturally integrated throughout the text.  The title and the feature descriptions are excellent.
*   **Clear Call to Actions (CTAs):** Encourages users to get started (e.g., "Get Free API Key").
*   **Detailed Explanations:** Provides more context, especially for complex concepts like OAuth.
*   **Code Examples:** Improved readability of code snippets.
*   **Links to Key Resources:** All relevant links (docs, website, Discord, examples, Docker images, etc.) are readily available.
*   **Alt Text for Images:** Added `alt` attributes to all images for accessibility and SEO.
*   **Concise and Focused:**  The README is now more direct and informative, removing any unnecessary fluff.
*   **GitHub Link:**  Added a final clear link back to the original GitHub repository.