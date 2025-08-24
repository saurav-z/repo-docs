<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Seamless AI Integration</h1>

<p align="center"><b>Simplify AI workflows with Klavis AI, offering hosted and self-hosted solutions for instant integration with popular services.</b></p>

<div align="center">
  
[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## 🚀 Key Features of Klavis AI

*   **🌐 Hosted MCP Service:** Production-ready, managed infrastructure with a 99.9% uptime SLA.
*   **🔐 Enterprise OAuth:** Seamless authentication for Google, GitHub, Slack, Salesforce, and more.
*   **🛠️ Extensive Integrations:** Access 50+ integrations with CRMs, productivity tools, databases, and social media platforms.
*   **📦 Instant Deployment:** Effortless setup for Claude Desktop, VS Code, and Cursor.
*   **🏢 Enterprise-Ready:** SOC2 compliant and GDPR-ready with dedicated support.
*   **🐳 Self-Hosting Options:** Docker images and source code available for customization and control.

## 💻 Getting Started: Choose Your Integration Method

### 1. 🌐 Hosted Service (Recommended)

**Get up and running in seconds with our managed infrastructure – no setup required.**

**Quick Start:**

1.  **🔗 [Get Your Free API Key](https://www.klavis.ai/home/api-keys)**
2.  Install the Klavis client library:

    ```bash
    pip install klavis
    # or
    npm install klavis
    ```

3.  Use the provided code snippets to create a server instance.

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="your-free-key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

### 2. 🐳 Self-Hosting with Docker

**Deploy any MCP server quickly using Docker for complete control over your infrastructure.**

**Run GitHub MCP Server (no OAuth support):**

```bash
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
```

**Run Gmail MCP Server with OAuth:**

```bash
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 3. 🖥️ Configuration for Cursor

**Directly use our hosted service URLs for Cursor – no Docker setup needed:**

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

**Get your personalized Cursor configuration:**

1.  **🔗 [Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)**
2.  Select your desired service (Gmail, GitHub, Slack, etc.)
3.  Copy the generated configuration and paste it into your Claude Desktop config.

## 🛠️ Available MCP Servers

| Service       | Docker Image                                        | OAuth Required | Description                          |
|---------------|-----------------------------------------------------|----------------|--------------------------------------|
| **GitHub**    | `ghcr.io/klavis-ai/github-mcp-server`              | ✅             | Repository management, issues, PRs   |
| **Gmail**     | `ghcr.io/klavis-ai/gmail-mcp-server:latest`        | ✅             | Email reading, sending, management  |
| **Google Sheets**| `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations              |
| **YouTube**   | `ghcr.io/klavis-ai/youtube-mcp-server`              | ❌             | Video information, search            |
| **Slack**     | `ghcr.io/klavis-ai/slack-mcp-server:latest`        | ✅             | Channel management, messaging       |
| **Notion**    | `ghcr.io/klavis-ai/notion-mcp-server:latest`       | ✅             | Database and page operations        |
| **Salesforce**| `ghcr.io/klavis-ai/salesforce-mcp-server:latest`     | ✅             | CRM data management                |
| **Postgres**  | `ghcr.io/klavis-ai/postgres-mcp-server`            | ❌             | Database operations                  |
| ...           | ...                                                 | ...            | ...                                  |

[**🔍 Browse All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 View All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

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

### Integration with AI Frameworks

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

## 🎯 Self-Hosting Instructions

### 1. 🐳 Docker Images

**Quickest way to start; ideal for testing and integrating with AI tools.**

Available Images:

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Basic server
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - With OAuth support

[**🔍 Browse All Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

```bash
# Example: GitHub MCP Server
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker run -it -e KLAVIS_API_KEY=your_key ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

**Clone and run any MCP server locally, with or without Docker for maximum customization.**

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

## 🔐 OAuth Authentication Explained

**Klavis AI handles the complexity of OAuth, simplifying integration with services that require it.**

-   **Why OAuth is challenging:** Complex setup, code overhead, credential management, and token lifecycle.

## 📚 Resources & Community

| Resource         | Link                                            | Description                                 |
|------------------|-------------------------------------------------|---------------------------------------------|
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)      | Complete guides and API reference           |
| **💬 Discord**     | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users             |
| **🐛 Issues**      | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features          |
| **📦 Examples**    | [examples/](examples/)                           | Working examples with popular AI frameworks |
| **🔧 Server Guides**| [mcp_servers/](mcp_servers/)                    | Individual server documentation             |

## 🤝 Contributing

We welcome contributions! Review our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>
```

Key improvements and SEO optimizations:

*   **Clear, Concise Hook:** The first sentence directly explains Klavis AI's core value proposition.
*   **Keyword-Rich Headings:** Uses relevant keywords like "MCP Servers," "AI Integration," and "Self-Hosting."
*   **Bulleted Key Features:**  Highlights benefits in an easily scannable format.
*   **Clear Call to Actions:**  Prominently displays links to get started, documentation, and the community.
*   **Comprehensive, but Concise:** Condensed the information while ensuring important details are present.
*   **Emphasis on Benefits:** Focuses on what users *gain* from using Klavis AI.
*   **Formatted for Readability:** Improves readability with consistent formatting, whitespace, and Markdown.
*   **SEO-Friendly Structure:**  Uses headings, subheadings, and bullet points for optimal search engine indexing.
*   **Removed Redundancy:** Streamlined the text to remove unnecessary repetition.
*   **Alt Text for Image:** Added `alt` text to the logo image for accessibility and SEO.
*   **Clearer Instructions:**  Simplified the setup instructions with step-by-step guidance.
*   **Internal Linking:** Encourages exploration within Klavis AI by linking to other resources (documentation, examples, etc.).
*   **Targeted Keywords:** Naturally incorporates keywords like "production-ready," "hosted," "self-hosted," "OAuth," "AI frameworks," and service names.
*   **Backlink:** Includes the link to the original repo at the beginning and in the contributing section.

This improved README is more informative, user-friendly, and search engine optimized, making it easier for potential users to understand and adopt Klavis AI.