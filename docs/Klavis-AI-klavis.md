<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Unlock the Power of MCP Servers for Your AI Applications</h1>

<p align="center"><strong>Easily integrate 50+ MCP servers for instant access to a wide range of tools, from Gmail to GitHub, and more, all with a simple API.</strong></p>

<div align="center">
  <a href="https://docs.klavis.ai">
    <img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation Badge">
  </a>
  <a href="https://www.klavis.ai">
    <img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website Badge">
  </a>
  <a href="https://discord.gg/p7TuTEcssn">
    <img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord Badge">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License Badge">
  </a>
  <a href="https://github.com/orgs/klavis-ai/packages">
    <img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Badge">
  </a>
</div>

## Key Features of Klavis AI

*   🌐 **Hosted Service**: Production-ready, managed infrastructure with 99.9% uptime.
*   🔐 **Enterprise OAuth**: Seamless authentication for popular services like Google, GitHub, and Slack.
*   🛠️ **50+ Integrations**: Access a wide array of tools including CRMs, productivity apps, databases, and social media.
*   🚀 **Instant Deployment**: Quick setup for tools like Claude Desktop, VS Code, and Cursor.
*   🏢 **Enterprise Ready**: SOC2 and GDPR compliant, with dedicated support.
*   📖 **Open Source**: Full source code available for customization and self-hosting ([Original Repo](https://github.com/Klavis-AI/klavis)).

## Getting Started with Klavis AI: Your Choice of Deployment

Choose your preferred method to get started quickly:

### 1.  🚀 **Quick Start: Hosted Service (Recommended)**

   **Get instant access to 50+ MCP servers with our managed infrastructure - no setup required!**

   *   **Benefits:** Simplest setup, handles OAuth, and requires zero infrastructure management.
   *   **How to Use:**

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
   *   🔗 **[Get Free API Key →](https://www.klavis.ai/home/api-keys)**

### 2.  🐳 **Self-Hosting with Docker**

   *   **Ideal for:** Running servers locally or integrating with AI tools.
   *   **How to Use:**

      ```bash
      # Run GitHub MCP Server (no OAuth support)
      docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
        ghcr.io/klavis-ai/github-mcp-server:latest

      # Run Gmail MCP Server with OAuth
      docker run -p 8000:5000 -e KLAVIS_API_KEY=your_key \
        ghcr.io/klavis-ai/gmail-mcp-server:latest
      ```
   *   **Cursor Configuration:** Use hosted service URLs directly:

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
      *   **Get Personalized Configuration:**
          1.  🔗 [Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)
          2.  Select a service (Gmail, GitHub, etc.)
          3.  Copy the generated configuration
          4.  Paste into your tool's configuration (e.g., Claude Desktop)

### 3.  🏗️ **Build from Source (Advanced)**

   *   **Ideal for:** Customization and deeper integration.

   1.  Clone the repository:

      ```bash
      git clone https://github.com/klavis-ai/klavis.git
      cd klavis/mcp_servers/github
      ```

   2.  Choose a build method (Docker or direct):

      *   **Using Docker:**

         ```bash
         docker build -t github-mcp .
         docker run -p 5000:5000 github-mcp
         ```

      *   **Run Directly (Go example):**

         ```bash
         go mod download
         go run server.go
         ```

      *   **Python Servers Example:**

         ```bash
         cd ../youtube
         pip install -r requirements.txt
         python server.py
         ```
      *   **Node.js Servers Example:**

         ```bash
         cd ../slack
         npm install
         npm start
         ```

   *   **Important:** Each server has setup instructions in its respective `README`.

## 🛠️ Available MCP Servers

| Service         | Docker Image                                       | OAuth Required | Description                             |
|-----------------|----------------------------------------------------|----------------|-----------------------------------------|
| **GitHub**       | `ghcr.io/klavis-ai/github-mcp-server`              | ✅             | Repository management, issues, PRs      |
| **Gmail**        | `ghcr.io/klavis-ai/gmail-mcp-server:latest`        | ✅             | Email reading, sending, management       |
| **Google Sheets**| `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                   |
| **YouTube**      | `ghcr.io/klavis-ai/youtube-mcp-server`             | ❌             | Video information, search                |
| **Slack**        | `ghcr.io/klavis-ai/slack-mcp-server:latest`        | ✅             | Channel management, messaging          |
| **Notion**       | `ghcr.io/klavis-ai/notion-mcp-server:latest`       | ✅             | Database and page operations           |
| **Salesforce**   | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`   | ✅             | CRM data management                    |
| **Postgres**     | `ghcr.io/klavis-ai/postgres-mcp-server`            | ❌             | Database operations                    |
| ...             | ...                                                | ...            | ...                                     |

*   [**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

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

### Integration with AI Frameworks:  OpenAI Function Calling Example

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

*   [**📖 View Complete Examples →**](examples/)

## 🌐 Hosted MCP Service - Zero Setup Required

**Get started instantly without infrastructure headaches:**

### ✨ **Benefits of Our Hosted Service:**

*   🚀 **Instant Setup**: MCP server ready in under a minute.
*   🔐 **OAuth Handled**: Complex authentication is fully managed.
*   🏗️ **No Infrastructure**: Run on our secure, scalable cloud.
*   📈 **Auto-Scaling**: Seamless scaling from prototype to production.
*   🔄 **Always Updated**: Benefit from the latest server versions.
*   💰 **Cost-Effective**: Pay-as-you-go, with a free tier.

### 💻 **Quick Integration:**

```python
from klavis import Klavis

# Get started with an API key
klavis = Klavis(api_key="your-free-key")

# Create any MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id",
    platform_name="MyApp"
)

# Server is immediately ready
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

*   🔗 **[Get Free API Key →](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation →](https://docs.klavis.ai)**

## 🔐 OAuth Authentication Explained

**Klavis simplifies complex OAuth setup, allowing you to focus on using the servers.**

### Why OAuth Requires Implementation?

*   🔧 **Complex Setup**:  Requires setting up OAuth apps with service-specific configurations.
*   📝 **Implementation Overhead**: Involves handling OAuth 2.0 flows, token refresh, and secure storage.
*   🔑 **Credential Management**: Securely managing OAuth app secrets.
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and other error scenarios.

## 📚 Resources & Community

| Resource          | Link                                           | Description                         |
|-------------------|------------------------------------------------|-------------------------------------|
| **📖 Documentation** | [docs.klavis.ai](https://docs.klavis.ai)       | Guides and API reference            |
| **💬 Discord**      | [Join Community](https://discord.gg/p7TuTEcssn) | Get help, connect, and chat.      |
| **🐛 Issues**       | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)     | Report bugs & request features    |
| **📦 Examples**     | [examples/](examples/)                        | Working examples with AI frameworks |
| **🔧 Server Guides**| [mcp_servers/](mcp_servers/)                | Individual server documentation    |

## 🤝 Contributing

We welcome contributions! Learn how to contribute to Klavis by reading our [Contributing Guide](CONTRIBUTING.md).

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
Key improvements and SEO considerations:

*   **Compelling Headline:**  Focuses on benefits ("Unlock the Power") and keywords ("MCP Servers", "AI Applications").
*   **Concise Hook:**  Clear and benefit-oriented first sentence.
*   **Clear Structure:** Uses headings and subheadings for readability.
*   **Keyword Optimization:** Includes target keywords throughout (MCP servers, AI, integrations, OAuth).
*   **Call to Actions:**  Encourages users to "Get Free API Key," "Join," and "View All Servers."
*   **Bullet Points:** Easy-to-scan key features and benefits.
*   **Multiple Deployment Options:** Clear instructions for hosted, Docker, and build-from-source, catering to various user needs.
*   **SEO-Friendly Image Alt Text:** Added `alt` text to the logo image and badges.
*   **Internal Linking:** Links to key resources (docs, Discord, examples) and within the document.
*   **Clear Code Examples:**  Relevant code snippets for Python and TypeScript users.
*   **OAuth Explanation:** Explains the complexities of OAuth and the benefits of using Klavis.
*   **Contribution Guide Mention:**  Encourages community contributions.
*   **Complete and Organized:**  Covers all the information from the original README in a more accessible and user-friendly format.
*   **Formatted for Readability:**  Uses Markdown effectively for emphasis and lists.
*   **Contextual URLs:** Clear links to key areas of Klavis resources.
*   **Footer with relevant links**.