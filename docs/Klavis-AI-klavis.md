<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unlock Seamless Integrations for Your AI Applications</h1>

**Klavis AI provides production-ready MCP (Model Control Plane) servers, enabling effortless integration with 50+ services, from Gmail to GitHub, all accessible through a simple API.**

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features of Klavis AI

*   **🚀 Instant Integration**: Connect to services in seconds with zero-configuration setup.
*   **🌐 Hosted and Self-Hosted Options**: Choose the right deployment model for your needs.
*   **🔐 Enterprise-Grade Security**:  Includes OAuth support, and SOC2 & GDPR compliance.
*   **🛠️ Wide Range of Integrations**: Access 50+ ready-to-use MCP servers, including Gmail, GitHub, and more.
*   **📦 Simple API Access**: Easy integration with Python, TypeScript, and other frameworks.
*   **🏢 Scalable Infrastructure**:  Production-ready managed infrastructure with 99.9% uptime SLA available.

## Getting Started: Choose Your Integration Method

### 1. 🌐 Hosted Service (Recommended for Production)

Leverage Klavis AI's fully managed infrastructure for a hassle-free experience.

**Benefits:**

*   **No setup required**: Start using servers instantly.
*   **Scalable**: Handles traffic automatically.
*   **Managed OAuth**: Authentication is simplified.

**Quick Start:**

1.  **🔗 [Get Free API Key](https://www.klavis.ai/home/api-keys)**

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

### 2. 🐳 Docker (Self-Hosting)

Deploy MCP servers directly to your infrastructure using Docker.

**How to Run:**

```bash
# Run GitHub MCP Server (no OAuth support)
docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
  ghcr.io/klavis-ai/github-mcp-server:latest

# Run Gmail MCP Server with OAuth
docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 3. 🖥️ Cursor Configuration

Integrate Klavis AI with Cursor by using our hosted service URLs directly:

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

**Get your configuration:**

1.  **🔗 [Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)**
2.  Select a service (Gmail, GitHub, etc.).
3.  Copy the generated configuration and paste it into your tool's config.

## 🛠️ Available MCP Servers

| Service          | Docker Image                                    | OAuth Required | Description                                |
| ---------------- | ----------------------------------------------- | -------------- | ------------------------------------------ |
| **GitHub**       | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs           |
| **Gmail**        | `ghcr.io/klavis-ai/gmail-mcp-server:latest`      | ✅             | Email reading, sending, management         |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                     |
| **YouTube**      | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌             | Video information, search                  |
| **Slack**        | `ghcr.io/klavis-ai/slack-mcp-server:latest`      | ✅             | Channel management, messaging              |
| **Notion**       | `ghcr.io/klavis-ai/notion-mcp-server:latest`     | ✅             | Database and page operations               |
| **Salesforce**   | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                      |
| **Postgres**     | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌             | Database operations                      |
| ...              | ...                                             | ...            | ...                                        |

**[🔍 View All 50+ Servers →](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [🐳 Browse Docker Images →](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)**

## 🎯 Advanced Usage Examples

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

### With AI Frameworks (OpenAI)

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

**[📖 View Complete Examples →](examples/)**

## 🔑 OAuth Authentication

Klavis simplifies OAuth authentication, handling complex setup automatically. Some servers require OAuth (Google, GitHub, Slack, etc.).  Klavis handles the complexities of the OAuth flow.

```bash
# Run with OAuth support (requires free API key)
docker run -it -e KLAVIS_API_KEY=your_free_key \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

## 📚 Resources and Community

*   **📖 Documentation**: [docs.klavis.ai](https://docs.klavis.ai)
*   **💬 Discord**: [Join Community](https://discord.gg/p7TuTEcssn)
*   **🐛 Issues**: [GitHub Issues](https://github.com/klavis-ai/klavis/issues)
*   **📦 Examples**: [examples/](examples/)
*   **🔧 Server Guides**: [mcp_servers/](mcp_servers/)

## 🤝 Contribute

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI! </strong></p>
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

*   **Clear, Concise Hook:** The one-sentence hook provides a clear value proposition.
*   **Keyword-Rich Headings:**  Uses relevant keywords like "MCP Servers," "AI Applications," "Integration," and service names.
*   **Bulleted Key Features:**  Easy-to-scan list highlighting benefits.
*   **Actionable Subheadings:** Guides users on how to start.
*   **Call to Action (CTA):** Encourages users to get started with a free API key.
*   **Internal Links:**  Links within the README to other sections and resources.
*   **External Links:**  Links to the website, documentation, Discord, and GitHub repo are prominent.
*   **Alt Text:**  Added alt text to the Klavis AI logo for accessibility and SEO.
*   **Reorganized for Clarity:** Improved the flow and structure.
*   **SEO Optimized Content:** The content is written with an eye towards relevant search terms.
*   **Summarized Content:**  The README is more concise, focusing on the most important information.
*   **Repository Link Added:** The original repo link is included.