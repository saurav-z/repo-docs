<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers</h1>
<p align="center"><strong>Seamlessly integrate your AI applications with 50+ services using Klavis AI's powerful MCP servers.</strong></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features of Klavis AI

*   ✅ **50+ Integrations:** Connect your AI with popular services like Gmail, GitHub, Slack, Salesforce, and more.
*   🐳 **Self-Hosted & Hosted Options:** Deploy MCP servers with Docker or leverage our fully managed service.
*   🔑 **Enterprise OAuth Support:** Simplify authentication with built-in support for Google, GitHub, Slack, and other OAuth-enabled services.
*   🚀 **Instant Deployment:** Get up and running quickly with zero-configuration setup for tools like Claude Desktop, VS Code, and Cursor.
*   🌐 **Production-Ready Infrastructure:** Benefit from our hosted service with a 99.9% uptime SLA and automatic scaling.
*   🏢 **Enterprise-Grade Security:** SOC2 compliant and GDPR-ready, with dedicated support for your needs.
*   📖 **Open Source:** Customize and extend Klavis AI with the full source code available.

## Quick Start: Deploy an MCP Server in Minutes

Choose your preferred deployment method:

### 🐳 **Option 1: Self-Hosting with Docker**

Easily run MCP servers on your own infrastructure using Docker.

1.  **Get an API Key (for OAuth support):** [Get Free API Key](https://www.klavis.ai/home/api-keys)

    ```bash
    # Run GitHub MCP Server with OAuth
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
      ghcr.io/klavis-ai/github-mcp-server:latest
    ```

    ```bash
    # Run GitHub MCP Server (manual token)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
      ghcr.io/klavis-ai/github-mcp-server:latest
    ```

    **Note:** The MCP server runs on port 5000, exposing the MCP protocol at `/mcp`.
    Example integration in Cursor:
    ```json
    {
      "mcpServers": {
        "github": {
          "url": "http://localhost:5000/mcp/"
        }
      }
    }
    ```

### 🌐 **Option 2: Hosted Service (Recommended)**

**Simplify your workflow with our managed infrastructure – no setup required.**

1.  **Get a Free API Key:** [Get Free API Key](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis client:**

    ```bash
    pip install klavis
    # or
    npm install klavis
    ```
3.  **Integrate with your code:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

    Example in Cursor:

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

**💡 Get your personalized configuration instantly:**

1.  **🔗 [Visit our MCP Servers page](https://www.klavis.ai/home/mcp-servers)**
2.  **Select any service** (Gmail, GitHub, Slack, etc.)
3.  **Copy the generated configuration**
4.  **Paste into your tool (e.g., Claude Desktop)**

## 🎯 Self-Hosting Instructions (Detailed)

### 1. 🐳 **Docker Images**

Ideal for testing or integrating with tools like Claude Desktop.

**Available Images:**
- `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support
- `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server build by selected commit ID

**[🔍 Browse All Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)**

```bash
# Example: GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Example: Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ **Build from Source**

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

## 🛠️ Available MCP Servers (Examples)

| Service         | Docker Image                                    | OAuth Required | Description                    |
| --------------- | ----------------------------------------------- | -------------- | ------------------------------ |
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`           | ✅             | Repository management, issues, PRs |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`       | ✅             | Email reading, sending, management |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations         |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌             | Video information, search      |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`       | ✅             | Channel management, messaging    |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`      | ✅             | Database and page operations    |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`  | ✅             | CRM data management           |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌             | Database operations            |
| ...             | ...                                             | ...            | ...                            |

And many more!
[**🔍 View All 50+ Servers**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

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

[**📖 View Complete Examples**](examples/)

## 🌐 Hosted MCP Service - Zero Setup Required

**Ideal for individuals and businesses seeking instant access without infrastructure overhead.**

### ✨ **Why Choose Our Hosted Service:**

*   🚀 **Instant Setup:** Get any MCP server running in seconds.
*   🔐 **OAuth Handled:** We manage the complex authentication process.
*   🏗️ **No Infrastructure:** Run everything on our secure and scalable cloud.
*   📈 **Auto-Scaling:** Easily scale from prototype to production.
*   🔄 **Always Updated:** Access the latest MCP server versions automatically.
*   💰 **Cost-Effective:** Pay only for the resources you use, with a free tier available.

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

**🔗 [Get Free API Key](https://www.klavis.ai/home/api-keys)** | **📖 [Complete Documentation](https://docs.klavis.ai)**

## 🔐 OAuth Authentication Explained

Klavis AI simplifies OAuth, the process for authenticating with services like Google, GitHub, and Slack.  OAuth implementation involves complex setup and code.

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth needs additional implementation?**
- 🔧 **Complex Setup**: Each service requires creating OAuth apps with specific redirect URLs, scopes, and credentials
- 📝 **Implementation Overhead**: OAuth 2.0 flow requires callback handling, token refresh, and secure storage
- 🔑 **Credential Management**: Managing multiple OAuth app secrets across different services
- 🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases

Our OAuth wrapper simplifies this by handling all the complex OAuth implementation details, so you can focus on using the MCP servers directly.

**Alternative**: For advanced users, you can implement OAuth yourself by creating apps with each service provider. Check individual server READMEs for technical details.

## 📚 Resources & Community

| Resource             | Link                                         | Description                                  |
| -------------------- | -------------------------------------------- | -------------------------------------------- |
| **📖 Documentation**  | [docs.klavis.ai](https://docs.klavis.ai)      | Complete guides and API reference            |
| **💬 Discord**      | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with other users       |
| **🐛 Issues**       | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features           |
| **📦 Examples**       | [examples/](examples/)                        | Working examples with popular AI frameworks  |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                  | Individual server documentation              |

## 🤝 Contributing

We welcome contributions! Help us improve Klavis AI by:
*   🐛 Reporting bugs and requesting features
*   📝 Improving documentation
*   🔧 Building new MCP servers
*   🎨 Enhancing existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to learn more.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI! </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">View on GitHub</a>
  </p>
</div>
```
Key improvements:

*   **SEO Optimization:**  Includes relevant keywords like "MCP servers," "AI integrations," "OAuth," and service names.  Uses headings for clarity and scannability.  Uses the "supercharge" language to be more engaging.
*   **One-Sentence Hook:**  Provides a clear value proposition at the beginning.
*   **Clear Structure:** Organizes content with headings and subheadings, making it easy to navigate.
*   **Bulleted Key Features:** Highlights the main benefits of using Klavis AI.
*   **Concise Explanations:**  Explains concepts without unnecessary jargon.
*   **Calls to Action:** Encourages users to get an API key, view documentation, and join the community.
*   **Complete Examples:** Shows how to use both hosted and self-hosted options.
*   **Clear Instructions:** Provides step-by-step guides for different use cases.
*   **Added GitHub Link:** Added the GitHub link to the final call to action.
*   **Emphasis on Key Benefits:** Uses bolding to highlight essential information.
*   **OAuth Explanation:** Provides a dedicated section to explain OAuth and the benefits of using Klavis AI's implementation.
*   **Cleaned Up Docker Commands:**  Simplified and clarified Docker commands.
*   **More Comprehensive Table:**  Expanded the MCP server table to include more details.
*   **Modernized Language:** Updated language for better readability.