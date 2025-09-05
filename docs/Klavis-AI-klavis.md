<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unleash the Power of Production-Ready MCP Servers</h1>

<p align="center"><strong>Quickly integrate 50+ services, from Gmail to GitHub, into your AI applications with Klavis AI's self-hosted and hosted solutions.</strong></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features:

*   **🚀 Instant Deployment:**  Deploy any MCP server in seconds, perfect for rapid integration with AI tools like Claude Desktop and VS Code.
*   **🌐 Hosted Service:**  Benefit from a production-ready, managed infrastructure with a 99.9% uptime SLA, eliminating setup complexities.
*   **🐳 Self-Hosting Options:**  Leverage Docker images and source code to customize and control your MCP servers.
*   **🔐 Enterprise OAuth:** Simplify authentication for services like Google, GitHub, and Slack with our built-in OAuth support.
*   **🛠️ Extensive Integrations:** Connect to 50+ services, including CRM, productivity tools, databases, and social media platforms.
*   **💰 Cost-Effective:**  Pay only for what you use, with a generous free tier available to get you started.
*   **📖 Open Source:** Access the full source code for customization and self-hosting.

## Getting Started: Run an MCP Server in 30 Seconds!

Klavis AI offers both self-hosted and hosted options to fit your needs.

### 🐳 Self-Hosting with Docker

Easily deploy MCP servers using Docker for local development and control.

1.  **Get Your Free API Key:**  [Get Free API Key](https://www.klavis.ai/home/api-keys) (Required for some servers)
2.  **Pull a Docker Image:**

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/github-mcp-server:latest
```

**Note:** The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

3.  **Example: Using in Cursor**:
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

Our hosted service provides a production-ready, managed environment, removing the hassle of setup and maintenance.

1.  **Get Your Free API Key:**  [Get Free API Key](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis client:**
    ```bash
    pip install klavis
    # or
    npm install klavis
    ```
3.  **Quick Integration Example:**

```python
from klavis import Klavis

klavis = Klavis(api_key="Your-Klavis-API-Key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```
4.  **Example running in Cursor:**

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

**💡 Configure Your Tools in Seconds:**

1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
2.  **Select a Service:**  Gmail, GitHub, Slack, and more.
3.  **Copy the Configuration:**  The generated configuration is tailored for your AI tool.
4.  **Paste & Integrate:**  Paste it into your AI application's configuration (e.g., Claude Desktop).

## 🎯 Self-Hosting Instructions: Dive Deeper

### 1. 🐳 Docker Images (Fastest Option)

Ready-to-use Docker images simplify self-hosting, ideal for integrating with AI tools.

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

### 2. 🏗️ Build from Source: Customization at Your Fingertips

Clone the repository and build any MCP server locally.

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

Check the individual README files within each server directory for detailed setup instructions.

**Or use our managed infrastructure - no Docker required:**
```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers: Your Integration Toolkit

| Service | Docker Image | OAuth Required | Description |
|---------|--------------|----------------|-------------|
| **GitHub** | `ghcr.io/klavis-ai/github-mcp-server` | ✅ | Repository management, issues, PRs |
| **Gmail** | `ghcr.io/klavis-ai/gmail-mcp-server:latest` | ✅ | Email reading, sending, management |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅ | Spreadsheet operations |
| **YouTube** | `ghcr.io/klavis-ai/youtube-mcp-server` | ❌ | Video information, search |
| **Slack** | `ghcr.io/klavis-ai/slack-mcp-server:latest` | ✅ | Channel management, messaging |
| **Notion** | `ghcr.io/klavis-ai/notion-mcp-server:latest` | ✅ | Database and page operations |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅ | CRM data management |
| **Postgres** | `ghcr.io/klavis-ai/postgres-mcp-server` | ❌ | Database operations |
| ... | ... | ...| ... |

And many more!

[**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples: Seamless Integration

### With AI Frameworks

**Python**

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

**TypeScript**

```typescript
import { KlavisClient } from 'klavis';

const klavis = new KlavisClient({ apiKey: 'your-key' });
const server = await klavis.mcpServer.createServerInstance({
    serverName: "Gmail",
    userId: "user123"
});
```

[**📖 View Complete Examples →**](examples/)

## 🌐 Hosted MCP Service: Effortless AI Integration

**Ideal for individuals and businesses seeking instant access without infrastructure complexities.**

### ✨ **Advantages of Our Hosted Service:**

*   **🚀 Instant Setup:** Get any MCP server running in seconds.
*   **🔐 Simplified OAuth:** We handle the complex authentication process.
*   **🏗️ Zero Infrastructure:** Runs on our secure and scalable cloud.
*   **📈 Automatic Scaling:** Scales seamlessly from prototype to production.
*   **🔄 Always Updated:** Receive the latest MCP server versions automatically.
*   **💰 Cost-Effective:** Enjoy a cost-efficient pay-as-you-go model with a free tier.

### 💻 **Rapid Integration:**

```python
from klavis import Klavis

# Start with an API key
klavis = Klavis(api_key="Your-Klavis-API-Key")

# Instantiate any MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

# The server is immediately ready for use
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication: Simplified

**Simplifying authentication for Google, GitHub, Slack, etc. Our implementation handles complex OAuth workflows.**

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Key Challenges of Implementing OAuth:**

*   🔧 **Complex Setup:** Requires OAuth app creation, redirect URLs, scopes, and credential management.
*   📝 **Implementation Overhead:** Involves handling callbacks, token refresh, and secure storage.
*   🔑 **Credential Management:** Managing OAuth secrets across multiple services can be challenging.
*   🔄 **Token Lifecycle:** Handling token expiration, refresh, and error conditions adds complexity.

Our OAuth wrapper simplifies this process, enabling you to concentrate on utilizing the MCP servers directly.

**Advanced Users:**  Explore the creation of apps with service providers for a custom OAuth implementation. See individual server READMEs for details.

## 📚 Resources & Community: Stay Connected

| Resource | Link | Description |
|----------|------|-------------|
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai) | Complete guides and API reference |
| **💬 Discord** | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users |
| **🐛 Issues** | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features |
| **📦 Examples** | [examples/](examples/) | Working examples with popular AI frameworks |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/) | Individual server documentation |

## 🤝 Contributing:  Join Us!

We encourage your contributions! Whether you'd like to:

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

Read our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📜 License

MIT License - See [LICENSE](LICENSE) for full details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications Today!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/klavis-ai/klavis">GitHub Repository</a>
  </p>
</div>
```
Key improvements and SEO optimizations:

*   **Strong Hook:**  A compelling one-sentence introduction that highlights the main value proposition.
*   **Clear Headings:**  Uses clear and descriptive headings for better readability and SEO.
*   **Bulleted Key Features:**  Highlights the benefits using bullet points, improving scannability.
*   **Keyword Optimization:**  Includes relevant keywords such as "MCP servers," "AI integration," "self-hosting," "hosted service," "OAuth," and service names (Gmail, GitHub, etc.).
*   **Concise Language:**  Uses clear and concise language to convey information effectively.
*   **Call to Actions (CTAs):**  Includes multiple calls to action (e.g., "Get Free API Key," "Documentation") to encourage user engagement.
*   **Internal Linking:** Links to internal resources like the documentation, Discord, and examples, enhancing user navigation and SEO.  Added a link back to the original GitHub repository.
*   **Schema-Friendly:**  Uses standard Markdown formatting.
*   **Mobile-Friendly:**  The formatting and structure are designed to be easily readable on any device.
*   **Focused Benefits:** Concentrates on *why* the user should use Klavis AI, not just *what* it is.
*   **Organization:** Improved the flow of information for a better user experience.
*   **Alt Text:** Added `alt` text to the logo image.
*   **Expanded Descriptions:**  Added more descriptive text to clarify the benefits and functionalities.