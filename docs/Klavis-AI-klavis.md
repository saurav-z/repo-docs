<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Seamless AI Integration</h1>

<p align="center"><strong>Unlock the power of AI with Klavis: Effortlessly integrate 50+ MCP servers for enhanced functionality.</strong></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

---

## Key Features

*   **🌐 Hosted Service:** Production-ready, managed infrastructure with 99.9% uptime.
*   **🔐 Enterprise OAuth:** Secure authentication for Google, GitHub, Slack, and more.
*   **🛠️ 50+ Integrations:** Connect to CRM, productivity tools, databases, and social media.
*   **🚀 Instant Deployment:** Zero-config setup for popular tools like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise Ready:** SOC2 compliant, GDPR ready, and backed by dedicated support.
*   **🐳 Self-Hosting Options:** Docker images and source code for customization.

---

## Quick Start: Access MCP Servers in Minutes

Klavis offers both a hosted service for ease of use and self-hosting options for greater control.

### 1.  Hosted Service (Recommended for Production)

**Get started with instant access to a wide range of MCP servers – no setup required!**

*   **Fastest Setup:** Simply obtain your API key.
*   **No Infrastructure Management:** We handle everything.
*   **Automatic Updates:** Benefit from the latest server versions.

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

**🔗 [Get Your Free API Key Now](https://www.klavis.ai/home/api-keys)**

### 2. Self-Hosting with Docker

**Deploy MCP servers quickly using Docker for complete control.**

```bash
# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 3. Cursor Configuration

**Seamlessly integrate Klavis MCP servers with Cursor.**

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

**To configure Cursor:**

1.  **🔗 [Visit the MCP Servers Page](https://www.klavis.ai/home/mcp-servers)**
2.  **Select** the service you want to use.
3.  **Copy** the configuration snippet provided.
4.  **Paste** it into your Cursor configuration file.

---

## Self-Hosting Guide: Build and Run Locally

### 1. Docker Images

**The easiest way to get started is with pre-built Docker images.**

**Available Images:**
-   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
-   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth support

**Browse All Docker Images:** [https://github.com/orgs/Klavis-AI/packages?repo_name=klavis](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)**

### 2. Build From Source

**For advanced users, build and run MCP servers from the source code.**

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

*   **Each server has detailed setup instructions in its README.**
*   **Or use our managed infrastructure - no Docker required.**

```bash
pip install klavis  # or npm install klavis
```

---

## Available MCP Servers

| Service         | Docker Image                                     | OAuth Required | Description                                   |
| --------------- | ------------------------------------------------ | -------------- | --------------------------------------------- |
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`             | ✅             | Repository management, issues, PRs          |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`       | ✅             | Email reading, sending, management         |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                      |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`            | ❌             | Video information, search                   |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`       | ✅             | Channel management, messaging              |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`      | ✅             | Database and page operations                |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`  | ✅             | CRM data management                         |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`           | ❌             | Database operations                         |
| ...             | ...                                              | ...            | ...                                           |

**[View All 50+ Servers](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)  |  [Browse Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)**

---

## Usage Examples

**Integrate Klavis with your existing AI applications.**

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

### OpenAI Function Calling

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

**[View Complete Examples](examples/)**

---

## Hosted Service Benefits: Zero Setup Required

**Perfect for individuals and businesses seeking immediate AI integration without infrastructure complexities.**

### ✨ Why Choose Our Hosted Service?

*   **🚀 Instant Setup:** Get any MCP server running in 30 seconds.
*   **🔐 OAuth Handled:** We manage the complexities of authentication.
*   **🏗️ No Infrastructure:** Run on our secure, scalable cloud.
*   **📈 Auto-Scaling:** Scale seamlessly from prototype to production.
*   **🔄 Always Updated:** Benefit from the latest server versions automatically.
*   **💰 Cost-Effective:** Pay only for what you use with a free tier available.

### 💻 Quick Integration:

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

**🔗 [Get Free API Key](https://www.klavis.ai/home/api-keys) | [Complete Documentation](https://docs.klavis.ai)**

---

## OAuth Authentication: Simplified

**Klavis simplifies the complexities of OAuth, enabling you to focus on your application.**

*   **Automated Setup:**  We handle the creation of OAuth apps with necessary credentials and scopes.
*   **Simplified Flows:** We manage the entire OAuth 2.0 flow, including handling callbacks, token refreshes, and error conditions.
*   **Secure Token Management:** We securely store and manage the lifecycle of access tokens.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

---

## Resources & Community

| Resource             | Link                                              | Description                                     |
| -------------------- | ------------------------------------------------- | ----------------------------------------------- |
| **📖 Documentation**  | [docs.klavis.ai](https://docs.klavis.ai)          | Complete guides and API reference               |
| **💬 Discord**        | [Join Community](https://discord.gg/p7TuTEcssn)   | Get help and connect with other users           |
| **🐛 Issues**         | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features              |
| **📦 Examples**      | [examples/](examples/)                              | Working examples with popular AI frameworks     |
| **🔧 Server Guides**  | [mcp_servers/](mcp_servers/)                        | Individual server documentation                 |

---

## Contributing

**We welcome your contributions to improve Klavis!**

*   🐛 Report bugs and suggest new features.
*   📝 Enhance our documentation.
*   🔧 Develop new MCP servers for additional integrations.
*   🎨 Improve the functionality of existing servers.

**See our [Contributing Guide](CONTRIBUTING.md) to learn how you can help!**

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repo</a>
  </p>
</div>
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:**  Uses strong keywords (Klavis AI, MCP Servers, AI Integration)
*   **Hook:** Starts with a compelling one-sentence description of Klavis's value.
*   **Headings & Structure:**  Organized with clear, descriptive headings (Quick Start, Self-Hosting, etc.) and subheadings for readability.
*   **Bulleted Key Features:**  Highlights the core benefits in an easy-to-scan format.
*   **Keyword Optimization:** Uses relevant keywords naturally throughout the content.
*   **Calls to Action:** Includes clear calls to action (Get Free API Key, View All Servers, Join Discord, etc.) to drive user engagement.
*   **Links:** Includes internal links to other sections of the README and external links to resources (website, docs, discord, examples, and importantly the GitHub repo itself, included at the end).
*   **Concise Language:** Avoids unnecessary jargon and keeps the information accessible.
*   **Emphasis on Benefits:** Focuses on the benefits of using Klavis (e.g., ease of use, time savings, enterprise-grade features).
*   **Updated formatting** Consistent styling for improved readability.
*   **Alt text**  Added `alt` text for images to improve accessibility.
*   **GitHub Repo Link**: Added a link back to the original repository at the end.

This revised README is much more user-friendly, SEO-optimized, and effective at conveying the value of Klavis to potential users.