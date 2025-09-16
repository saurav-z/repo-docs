<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI:  The Fastest Way to Integrate Any Tool into Your AI Applications</h1>

<p align="center"><strong>Unlock 50+ MCP Servers for Seamless AI Integration: Self-Hosted Solutions, Hosted MCP Service, and Enterprise-Grade OAuth.</strong></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## 🚀 Key Features of Klavis AI

*   **🌐 Hosted Service:** Ready-to-use managed infrastructure with 99.9% uptime.
*   **🐳 Self-Hosting Options:** Docker images and source code for flexibility.
*   **🔐 Enterprise OAuth:** Secure and simplified authentication for popular services.
*   **🛠️ 50+ Integrations:** Connect to Gmail, GitHub, Slack, and more.
*   **🚀 Instant Deployment:** Integrate with tools like Claude Desktop, VS Code, and Cursor with zero configuration.
*   **💡 OpenAI Function Calling:** Integrate with OpenAI, and other LLMs for easy development.

## 📦 Get Started: Run Any MCP Server in Seconds

Klavis AI offers multiple ways to get up and running with MCP (Message Channel Protocol) servers.  Choose the option that best fits your needs:

### 🐳 **1. Self-Hosting with Docker (For full control and customization)**

**Quickly deploy any MCP server using Docker.**  This is ideal for those who want to manage their own infrastructure or integrate with AI tools.

**Steps:**

1.  **Get Your API Key:**  [Get Free API Key →](https://www.klavis.ai/home/api-keys) (required for OAuth-enabled servers)

2.  **Pull the Docker Image:**
    ```bash
    # Example: GitHub MCP Server (OAuth support - requires API Key)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest

    # Or run Github MCP Server (manually add token)
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
    ```

3.  **Access the Server:** The MCP server will be running on port 5000, exposing the MCP protocol at the `/mcp` path.

   **Example Configuration (Cursor):**
   ```json
   {
     "mcpServers": {
       "github": {
         "url": "http://localhost:5000/mcp/"
       }
     }
   }
   ```

**Browse All Docker Images:** [🔍 Browse All Docker Images →](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

### 🌐 **2.  Hosted Service (Recommended for Production - No Setup Required)**

**Leverage our managed infrastructure for instant access to 50+ MCP servers.**  This is the fastest and easiest way to integrate with AI tools.

**Steps:**

1.  **Install the Klavis Client:**
    ```bash
    pip install klavis # Python
    # or
    npm install klavis # Node.js
    ```

2.  **Get Your API Key:**  [Get Free API Key →](https://www.klavis.ai/home/api-keys)

3.  **Create a Server Instance:**

    ```python
    from klavis import Klavis

    klavis = Klavis(api_key="Your-Klavis-API-Key")
    server = klavis.mcp_server.create_server_instance("GMAIL", "user123")
    ```

   **Example Configuration (Cursor):**

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

4.  **Configure your AI tool:**

    1.  **🔗 [Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
    2.  **Select any service** (Gmail, GitHub, Slack, etc.)
    3.  **Copy the generated configuration** for your tool
    4.  **Paste into Claude Desktop config** - done!

## ✨ Enterprise-Grade MCP Infrastructure

*   **🏢 Enterprise Ready:** SOC2 compliant, GDPR ready, with dedicated support.
*   **📖 Open Source**: Full source code available for customization and self-hosting.

## 🎯 Self Hosting Instructions

In addition to Docker, you can also build Klavis AI MCP servers from source:

### 🏗️ Build from Source

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/klavis-ai/klavis.git
    cd klavis/mcp_servers/github  # Or the directory of the server you want to run.
    ```

2.  **Choose Your Build Method:**

    *   **Option A: Using Docker (Recommended)**
        ```bash
        docker build -t github-mcp .
        docker run -p 5000:5000 github-mcp
        ```

    *   **Option B: Run directly (Go example)**
        ```bash
        go mod download
        go run server.go
        ```

    *   **Option C: Python servers**
        ```bash
        cd ../youtube
        pip install -r requirements.txt
        python server.py
        ```

    *   **Option D: Node.js servers**
        ```bash
        cd ../slack
        npm install
        npm start
        ```

**Important:** Each server's individual README within the `mcp_servers` directory contains detailed setup instructions specific to that service.

## 🛠️ Available MCP Servers - Connect to the Tools You Love

Klavis AI offers a wide range of MCP servers, with more being added regularly:

| Service        | Docker Image                                   | OAuth Required | Description                                     |
|----------------|------------------------------------------------|----------------|-------------------------------------------------|
| **GitHub**     | `ghcr.io/klavis-ai/github-mcp-server`          | ✅             | Repository management, issues, PRs             |
| **Gmail**      | `ghcr.io/klavis-ai/gmail-mcp-server:latest`    | ✅             | Email reading, sending, management           |
| **Google Sheets**| `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅           | Spreadsheet operations                        |
| **YouTube**    | `ghcr.io/klavis-ai/youtube-mcp-server`         | ❌             | Video information, search                     |
| **Slack**      | `ghcr.io/klavis-ai/slack-mcp-server:latest`    | ✅             | Channel management, messaging                 |
| **Notion**     | `ghcr.io/klavis-ai/notion-mcp-server:latest`   | ✅             | Database and page operations                  |
| **Salesforce** | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                         |
| **Postgres**   | `ghcr.io/klavis-ai/postgres-mcp-server`        | ❌             | Database operations                         |
| ...            | ...                                            | ...            | ...                                             |

**Explore the complete list:**  [**🔍 View All 50+ Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [**🐳 Browse Docker Images →**](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples - Integrate with Your Favorite Tools

Here's how to use Klavis AI with popular AI frameworks:

**Python Example**
```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123"
)
```

**TypeScript Example**
```typescript
import { KlavisClient } from 'klavis';

const klavis = new KlavisClient({ apiKey: 'your-key' });
const server = await klavis.mcpServer.createServerInstance({
    serverName: "Gmail",
    userId: "user123"
});
```

### Integrate with AI Frameworks

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

## 🌐 Hosted MCP Service - Simplify Your Workflow

**Our Hosted MCP Service provides a hassle-free experience for integrating AI with your favorite tools.**

### ✨ **Why Choose Our Hosted Service:**

*   **🚀 Instant Setup**: Get any MCP server running in 30 seconds.
*   **🔐 OAuth Handled**: We take care of the complex authentication.
*   **🏗️ No Infrastructure**: Run everything on our secure cloud.
*   **📈 Auto-Scaling**: Scale seamlessly from prototype to production.
*   **🔄 Always Updated**: Benefit from the latest MCP server versions.
*   **💰 Cost-Effective**: Pay only for what you use, with a free tier.

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

## 🔐 OAuth Authentication - Simplified

Klavis AI simplifies the OAuth authentication process, making it easier to connect to services like Google, GitHub, and Slack.

*   **Simplified Implementation:** Avoid complex setup and code.
*   **Automatic Handling:** We manage the details of authentication, refresh, and token management.

**Why Klavis simplifies OAuth:**
-   **🔧 Complex Setup**: Each service requires creating OAuth apps with specific redirect URLs, scopes, and credentials
-   **📝 Implementation Overhead**: OAuth 2.0 flow requires callback handling, token refresh, and secure storage
-   🔑 **Credential Management**: Managing multiple OAuth app secrets across different services
-   🔄 **Token Lifecycle**: Handling token expiration, refresh, and error cases

## 📚 Resources & Community - Get Support and Learn More

| Resource                  | Link                                           | Description                                               |
|---------------------------|------------------------------------------------|-----------------------------------------------------------|
| **📖 Documentation**      | [docs.klavis.ai](https://docs.klavis.ai)      | Complete guides and API reference                        |
| **💬 Discord**           | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with users                          |
| **🐛 Issues**             | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request features                          |
| **📦 Examples**           | [examples/](examples/)                           | Working examples with popular AI frameworks               |
| **🔧 Server Guides**      | [mcp_servers/](mcp_servers/)                   | Individual server documentation                           |

## 🤝 Contributing - Build the Future of AI Integration

We welcome contributions!  Help us make Klavis AI even better by:

*   🐛 Reporting bugs and requesting features
*   📝 Improving documentation
*   🔧 Building new MCP servers
*   🎨 Enhancing existing servers

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repository</a>
  </p>
</div>
```
Key improvements:

*   **SEO Optimization:** Added keywords like "AI integration," "MCP Servers," "OpenAI," "function calling," and tool names.
*   **Clear Headings:**  Organized content with clear, descriptive headings.
*   **Concise Summary:** A focused one-sentence hook to grab attention and explain the purpose.
*   **Bulleted Lists:**  Improved readability and highlighted key features.
*   **Emphasis on Benefits:** Focused on the value proposition (e.g., ease of use, speed).
*   **Actionable Steps:**  Provided clear, step-by-step instructions.
*   **Call to Action:** Included prominent calls to action to get a free API key and explore documentation.
*   **Consistent Formatting:** Used bolding and other formatting to improve readability.
*   **Links back to original repository:** Added a link to the original repo in the end.
*   **Concise content:** summarized content to its essential points.