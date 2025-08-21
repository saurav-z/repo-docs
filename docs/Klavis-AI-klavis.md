<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Applications</h1>

<p align="center"><b>Unlock the power of AI with Klavis AI's hosted and self-hosted MCP servers, offering seamless integrations, enterprise-grade features, and instant deployment.</b></p>

<div align="center">
  <a href="https://docs.klavis.ai"><img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation"></a>
  <a href="https://www.klavis.ai"><img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website"></a>
  <a href="https://discord.gg/p7TuTEcssn"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/orgs/klavis-ai/packages"><img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images"></a>
</div>

## Key Features

*   **🌐 Hosted Service:** Managed infrastructure with 99.9% uptime SLA, requiring zero setup.
*   **🐳 Self-Hosted Solutions:** Deploy MCP servers using Docker for complete control.
*   **🔐 Enterprise OAuth:** Secure authentication for popular services like Google, GitHub, and Slack.
*   **🛠️ 50+ Integrations:** Connect to CRMs, productivity tools, databases, and social media platforms.
*   **🚀 Instant Deployment:** Seamless integration with tools like Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise-Ready:** SOC2 compliant, GDPR ready, with dedicated support options.
*   **📖 Open Source:** Full source code is available for customization and self-hosting.

## 🚀 Quick Start: Deploy MCP Servers in Minutes

Get started with Klavis AI by using our hosted service or by self-hosting with Docker.

### 🌐 Hosted Service (Recommended)

Access a managed infrastructure with over 50 MCP servers and easily integrate with AI applications. No setup is required!

**Get your Free API Key:** [Get Free API Key →](https://www.klavis.ai/home/api-keys)

**Install the Klavis library:**

```bash
pip install klavis
# or
npm install klavis
```

**Example usage (Python):**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-free-key")
server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
```

### 🐳 Self-Hosting with Docker

Deploy MCP servers with ease, giving you full control over your infrastructure.

**Run a GitHub MCP Server:**

```bash
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest
```

**Run a Gmail MCP Server with OAuth:**

```bash
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

**Run a YouTube MCP Server:**

```bash
docker run -p 5000:5000 ghcr.io/klavis-ai/youtube-mcp-server:latest
```

### 🖥️ Cursor Configuration Example

Easily integrate Klavis AI with Cursor using the hosted service URLs:

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

1.  [Visit the MCP Servers page](https://www.klavis.ai/home/mcp-servers)
2.  Select your desired service (Gmail, GitHub, Slack, etc.)
3.  Copy the generated configuration
4.  Paste into your tool's configuration (e.g., Claude Desktop)

## 🛠️ Available MCP Servers

| Service        | Docker Image                                    | OAuth Required | Description                                |
| -------------- | ----------------------------------------------- | -------------- | ------------------------------------------ |
| GitHub         | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs         |
| Gmail          | `ghcr.io/klavis-ai/gmail-mcp-server:latest`      | ✅             | Email reading, sending, management         |
| Google Sheets  | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                      |
| YouTube        | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌             | Video information, search                  |
| Slack          | `ghcr.io/klavis-ai/slack-mcp-server:latest`      | ✅             | Channel management, messaging              |
| Notion         | `ghcr.io/klavis-ai/notion-mcp-server:latest`     | ✅             | Database and page operations               |
| Salesforce     | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                       |
| Postgres       | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌             | Database operations                       |
| ...            | ...                                             | ...            | ...                                        |

**Explore all 50+ servers:** [View All Servers](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [Browse Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

Integrate Klavis AI with your existing MCP implementations seamlessly:

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

### With AI Frameworks - OpenAI Example

**OpenAI Function Calling:**

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

[Complete Code Examples](examples/)

## 🌐 Hosted MCP Service - Zero Setup Required

Choose our hosted service for instant access without needing to manage infrastructure.

### ✨ Why Choose Our Hosted Service:

*   **🚀 Instant Setup**: Get any MCP server running in 30 seconds.
*   **🔐 OAuth Handled**: Complex authentication is handled by our platform.
*   **🏗️ No Infrastructure**: Everything runs on our secure, scalable cloud.
*   **📈 Auto-Scaling**: Scales seamlessly from prototype to production.
*   **🔄 Always Updated**: Always updated to the latest MCP server versions.
*   **💰 Cost-Effective**: Pay only for what you use; a free tier is available.

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

**🔗 [Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication Explained

Klavis AI simplifies OAuth authentication for services that require it.

**Why OAuth requires additional implementation:**

*   🔧 **Complex Setup**: Each service requires OAuth app creation with specific settings.
*   📝 **Implementation Overhead**: OAuth 2.0 flow involves handling redirects, token refreshes, and secure storage.
*   🔑 **Credential Management**: Managing multiple OAuth app secrets across different services.
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases.

Our OAuth wrapper simplifies this process, allowing you to focus on the use of MCP servers directly.

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images

The fastest way to get started is to use the Docker images available:

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic Server
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth Support

**Example: Running the GitHub MCP Server:**

```bash
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest
```

**Example: Gmail with OAuth (requires API key):**

```bash
docker run -it -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[Get Free API Key →](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Build and run any MCP server locally (with or without Docker):

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

Each server includes detailed setup instructions in its individual README file.

## 📚 Resources & Community

| Resource               | Link                                     | Description                                   |
| ---------------------- | ---------------------------------------- | --------------------------------------------- |
| **📖 Documentation**   | [docs.klavis.ai](https://docs.klavis.ai)  | Complete guides and API reference             |
| **💬 Discord**         | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users               |
| **🐛 Issues**          | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features           |
| **📦 Examples**        | [examples/](examples/)                   | Working examples with popular AI frameworks  |
| **🔧 Server Guides**   | [mcp_servers/](mcp_servers/)             | Individual server documentation               |

## 🤝 Contributing

We welcome contributions! Feel free to:

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  Includes keywords like "MCP Servers," "AI Applications," "Hosted," "Self-Hosted," and service names (Gmail, GitHub, etc.) throughout the README. Headings and subheadings are well-structured to aid readability and search indexing.
*   **One-Sentence Hook:** The opening paragraph provides a compelling introduction to the problem Klavis AI solves.
*   **Clear Structure:** Uses headings, subheadings, and bullet points for readability and ease of understanding.  The organization is improved, following a logical flow for onboarding new users.
*   **Call to Action:** Prominent "Get Free API Key" and links to documentation and other resources.
*   **Concise Language:** Streamlined descriptions and instructions for better user comprehension.
*   **Focus on Benefits:** Highlights the advantages of using Klavis AI (Hosted, Self-Hosted, Integrations, etc.).
*   **Complete Examples:**  Provides full code examples for both Python and TypeScript, including a more complete example with OpenAI integration.
*   **Clear Instructions:**  Offers step-by-step instructions for both hosted and self-hosted options.
*   **Community and Resources:** Includes a comprehensive list of useful links, including documentation, discord, examples, and a contributing guide.
*   **Emphasis on Hosted Service:** Prioritizes and clearly explains the ease of use of the hosted service.
*   **OAuth Section Enhanced:** Better explanations of the complexities of OAuth and how Klavis AI simplifies them.
*   **Alt Text for Images:** Adds `alt` text to the image tags for better accessibility and SEO.
*   **Complete Code Blocks:** Ensures all code blocks are formatted and easily copied.
*   **Link Back to Original Repo:** Maintained link to the original repo.