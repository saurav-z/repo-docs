<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for Seamless AI Integration</h1>

**Integrate AI with ease! Klavis AI provides self-hosted and hosted MCP (Message Control Protocol) servers for connecting your AI applications to various services like Gmail, GitHub, and more.**

<div align="center">
  <a href="https://docs.klavis.ai"><img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation"></a>
  <a href="https://www.klavis.ai"><img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website"></a>
  <a href="https://discord.gg/p7TuTEcssn"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/orgs/klavis-ai/packages"><img src="https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker" alt="Docker Images"></a>
</div>

## Key Features

*   ✅ **Hosted MCP Service:** Production-ready infrastructure with a 99.9% uptime SLA and automatic scaling.
*   🐳 **Self-Hosting Options:** Run MCP servers using Docker or build from source for complete control.
*   🔑 **Enterprise OAuth:** Secure and seamless authentication for services like Google, GitHub, and Slack.
*   ⚙️ **50+ Integrations:** Connect to a wide range of services, including CRM, productivity tools, databases, and social media.
*   🚀 **Instant Deployment:** Quick setup for popular AI tools like Claude Desktop, VS Code, and Cursor.
*   🛡️ **Enterprise-Ready:** SOC2 compliant, GDPR ready, and backed by dedicated support.
*   📖 **Open Source:** Customize and extend Klavis AI with access to full source code.

## 🚀 Quick Start: Run any MCP Server in Seconds!

Choose your preferred method:

### 🐳 Self-Hosting with Docker

1.  **Get Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Run a server:**

    ```bash
    # Example: Run Github MCP Server (requires API Key)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
      ghcr.io/klavis-ai/github-mcp-server:latest
    ```

    ```bash
    # Or run GitHub MCP Server (manually add token)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' \
      ghcr.io/klavis-ai/github-mcp-server:latest
    ```

    The MCP server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.

    **Example in Cursor:**

    ```json
    {
      "mcpServers": {
        "github": {
          "url": "http://localhost:5000/mcp/"
        }
      }
    }
    ```

### 🌐 Using Hosted Service (Recommended for Production)

Get started without any setup.

1.  **Get Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis library:**

    ```bash
    pip install klavis
    # or
    npm install klavis
    ```

3.  **Use the Klavis library:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

    **Example in Cursor:**

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

    **Get your personalized configuration in three steps:**

    1.  **Visit our MCP Servers page:** [https://www.klavis.ai/home/mcp-servers](https://www.klavis.ai/home/mcp-servers)
    2.  **Select your desired service** (e.g., Gmail, GitHub, Slack).
    3.  **Copy the generated configuration** and paste it into your tool's configuration.

## 🎯 Self Hosting Instructions

### 1. 🐳 Docker Images (Fastest Way to Start)

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
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest
```

[**🔗 Get Free API Key →**](https://www.klavis.ai/home/api-keys)

### 2. 🏗️ Build from Source

Clone and run any MCP server locally (with or without Docker):

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

Use our managed infrastructure - no Docker required:

```bash
pip install klavis  # or npm install klavis
```

## 🛠️ Available MCP Servers

| Service         | Docker Image                                    | OAuth Required | Description                                 |
|-----------------|-------------------------------------------------|----------------|---------------------------------------------|
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`           | ✅              | Repository management, issues, PRs        |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`     | ✅              | Email reading, sending, management          |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅              | Spreadsheet operations                      |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`          | ❌              | Video information, search                 |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`     | ✅              | Channel management, messaging             |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`    | ✅              | Database and page operations                |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅              | CRM data management                       |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`         | ❌              | Database operations                       |
| ...             | ...                                             | ...            | ...                                         |

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

**Perfect for individuals and businesses who want instant access without infrastructure complexity:**

### ✨ **Why Choose Our Hosted Service:**

*   🚀 **Instant Setup**: Get any MCP server running in 30 seconds
*   🔐 **OAuth Handled**: No complex authentication setup required
*   🏗️ **No Infrastructure**: Everything runs on our secure, scalable cloud
*   📈 **Auto-Scaling**: From prototype to production seamlessly
*   🔄 **Always Updated**: Latest MCP server versions automatically
*   💰 **Cost-Effective**: Pay only for what you use, free tier available

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

Some servers require OAuth authentication (Google, GitHub, Slack, etc.).

```bash
# Run with OAuth support (requires free API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY \
  ghcr.io/klavis-ai/gmail-mcp-server:latest

# Follow the displayed URL to authenticate
# Server starts automatically after authentication
```

**Why OAuth needs additional implementation?**

*   🔧 **Complex Setup**: Each service requires creating OAuth apps with specific redirect URLs, scopes, and credentials
*   📝 **Implementation Overhead**: OAuth 2.0 flow requires callback handling, token refresh, and secure storage
*   🔑 **Credential Management**: Managing multiple OAuth app secrets across different services
*   🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases

Our OAuth wrapper simplifies this by handling all the complex OAuth implementation details, so you can focus on using the MCP servers directly.

**Alternative**: For advanced users, you can implement OAuth yourself by creating apps with each service provider. Check individual server READMEs for technical details.

## 📚 Resources & Community

| Resource             | Link                                               | Description                                     |
|----------------------|----------------------------------------------------|-------------------------------------------------|
| **📖 Documentation**  | [docs.klavis.ai](https://docs.klavis.ai)           | Complete guides and API reference               |
| **💬 Discord**       | [Join Community](https://discord.gg/p7TuTEcssn)   | Get help and connect with users                 |
| **🐛 Issues**         | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features                |
| **📦 Examples**       | [examples/](examples/)                              | Working examples with popular AI frameworks      |
| **🔧 Server Guides** | [mcp_servers/](mcp_servers/)                      | Individual server documentation                  |

## 🤝 Contributing

We welcome contributions! To get involved:

*   🐛 Report bugs or request features
*   📝 Improve documentation
*   🔧 Build new MCP servers
*   🎨 Enhance existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to learn more!

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
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repository</a>
  </p>
</div>
```
Key changes and improvements:

*   **Stronger Hook:** Replaced the generic opening with a more benefit-driven hook: "Integrate AI with ease! Klavis AI provides self-hosted and hosted MCP (Message Control Protocol) servers for connecting your AI applications to various services like Gmail, GitHub, and more." This is much more effective for grabbing attention.
*   **SEO Optimization:**  Added keywords like "MCP servers," "AI integration," and specific services (Gmail, GitHub) throughout the text.
*   **Clear Headings and Structure:**  Organized the content with clear headings and subheadings for easy readability and scannability (crucial for SEO and user experience).
*   **Bulleted Key Features:**  Used bullet points to highlight the main benefits and features of Klavis AI, making them easy to digest.
*   **Concise Language:** Streamlined the text to be more direct and easier to understand.
*   **Actionable Quick Start:** Provided clear and concise instructions for both Docker and hosted service options. The API key link is included to encourage immediate use.
*   **Call to Actions:**  Included "Get Free API Key" links throughout to drive conversions.
*   **Comprehensive Resource Section:**  Provides direct links to documentation, Discord, and other useful resources.
*   **GitHub Repository Link:**  Added a direct link back to the original GitHub repository at the end to increase discoverability and engagement.
*   **Improved Formatting:**  Used bolding and other formatting to emphasize key information.
*   **Alt Text:**  Added alt text to images for better accessibility and SEO.
*   **Added Summary for hosted and self-hosting, in Quick Start**
*   **Removed Duplicates:** consolidated information where needed.
*   **Added Descriptions:** for Docker Images table.

This revised README is significantly more effective at attracting users, conveying the value of Klavis AI, and encouraging adoption.  It's also much better optimized for search engines.