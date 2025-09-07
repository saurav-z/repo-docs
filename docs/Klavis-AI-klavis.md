<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Production-Ready MCP Servers for AI Applications</h1>

<p align="center"><strong>Unlock 50+ MCP Servers with Self-Hosting, Managed Services & Enterprise OAuth.  Klavis AI empowers developers to easily integrate with popular services for their AI applications.</strong></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

## Key Features

*   **Self-Hosted Solutions:** Deploy MCP servers using Docker, providing flexibility and control.
*   **Hosted MCP Service:**  Access a managed infrastructure with 99.9% uptime and instant setup.
*   **Enterprise OAuth Support:**  Seamlessly integrate with Google, GitHub, Slack, and other services.
*   **50+ Integrations:**  Connect to CRM, productivity tools, databases, and social media.
*   **Instant Deployment:**  Get up and running with Claude Desktop, VS Code, and Cursor with zero config.
*   **Open Source:** Customize and self-host with open-source code.

## Getting Started with Klavis AI

Klavis AI offers both self-hosting options for those needing granular control and a fully managed service for rapid integration.

### 🐳 Self-Hosting with Docker

Easily run any MCP server in seconds using Docker. This is ideal for local development, testing, or integrating with AI tools like Claude Desktop.

1.  **Get Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Pull the Docker Image:**
    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    ```
3.  **Run the Docker Container:**
    ```bash
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
    ```
    *Or, use a GitHub token (if preferred)*
    ```bash
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
    ```
    *Remember that the server runs on port 5000 and exposes the MCP protocol at the `/mcp` path.*

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

### 🌐 Hosted Service (Recommended for Production)

Our managed infrastructure provides instant access to over 50 MCP servers with no setup required. Perfect for production environments.

1.  **Get Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys)
2.  **Install the Klavis Client:**
    ```bash
    pip install klavis  # or npm install klavis
    ```
3.  **Integrate into your code:**
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

**Quick Configuration:**

1.  **Browse MCP Servers:** [https://www.klavis.ai/home/mcp-servers](https://www.klavis.ai/home/mcp-servers)
2.  **Select a Service:** (Gmail, GitHub, Slack, etc.)
3.  **Copy the Configuration:** Paste the generated configuration into your AI tool.

## ✨ Benefits of Klavis AI

*   **Managed Infrastructure:** Reliable, scalable, and production-ready with a 99.9% uptime SLA.
*   **Enterprise Security:**  SOC2 compliant and GDPR ready.
*   **Dedicated Support:**  Access to professional support.
*   **Cost-Effective:** Pay only for the resources you use.
*   **Always Up-to-Date:**  Benefit from the latest MCP server versions automatically.

## 🎯 Self-Hosting Instructions (Detailed)

Klavis AI provides flexible self-hosting options using Docker and building from source code.

### 1. 🐳 Docker Images (Fastest Method)

Quickly test and integrate MCP servers using pre-built Docker images.

**Available Images:**

*   `ghcr.io/klavis-ai/{server-name}-mcp-server:latest` - Server with OAuth support.
*   `ghcr.io/klavis-ai/{server-name}-mcp-server:commit-id` - Server build by selected commit ID.

**Browse All Images:** [https://github.com/orgs/Klavis-AI/packages?repo_name=klavis](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

**Example Usage:**

```bash
# GitHub MCP Server
docker pull ghcr.io/klavis-ai/github-mcp-server:latest
docker run -p 5000:5000 ghcr.io/klavis-ai/github-mcp-server:latest

# Gmail with OAuth (requires API key)
docker pull ghcr.io/klavis-ai/gmail-mcp-server:latest
docker run -it -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/gmail-mcp-server:latest
```

### 2. 🏗️ Build from Source

Clone the repository and build any MCP server locally:

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

*Each server directory contains individual setup instructions.*

## 🛠️ Available MCP Servers (Partial List)

| Service         | Docker Image                                    | OAuth Required | Description                              |
|-----------------|-------------------------------------------------|----------------|------------------------------------------|
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`             | ✅             | Repository management, issues, PRs      |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`      | ✅             | Email reading, sending, management      |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                  |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`            | ❌             | Video information, search               |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`      | ✅             | Channel management, messaging         |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`     | ✅             | Database and page operations          |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest`   | ✅             | CRM data management                     |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`           | ❌             | Database operations                     |
| ...             | ...                                             | ...            | ...                                      |

**View All Servers:** [https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | **Browse Docker Images:** [https://github.com/orgs/Klavis-AI/packages?repo_name=klavis](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

## 💡 Usage Examples

Integrate Klavis AI into your existing AI applications.

**Python Example:**

```python
from klavis import Klavis

klavis = Klavis(api_key="your-key")
server = klavis.mcp_server.create_server_instance(
    server_name="YOUTUBE",
    user_id="user123"
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

### Using with AI Frameworks (OpenAI Example)

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

**View complete examples:** [examples/](examples/)

## 🌐 Hosted MCP Service - Simplified Integration

Our hosted service offers the fastest path to integrating MCP servers.  Perfect for those wanting to bypass the complexities of self-hosting.

### ✨ Advantages of the Hosted Service:

*   **Instant Setup**:  Get any MCP server running in seconds.
*   **Simplified Authentication**:  OAuth is handled for you.
*   **Zero Infrastructure**:  Benefit from a scalable, secure cloud infrastructure.
*   **Automatic Scaling**: From development to production with ease.
*   **Continuous Updates**:  Always access the latest MCP server versions.
*   **Cost-Effective**:  Pay only for what you use, with a free tier available.

### 💻 Quick Integration:

```python
from klavis import Klavis

# Get started with an API key
klavis = Klavis(api_key="Your-Klavis-API-Key")

# Create an MCP server instantly
gmail_server = klavis.mcp_server.create_server_instance(
    server_name="GMAIL",
    user_id="your-user-id"
)

# Server is ready
print(f"Gmail MCP server ready: {gmail_server.server_url}")
```

**Get a Free API Key:** [https://www.klavis.ai/home/api-keys](https://www.klavis.ai/home/api-keys) | **Complete Documentation:** [https://docs.klavis.ai](https://docs.klavis.ai)

## 🔐 OAuth Authentication Explained

Klavis AI simplifies OAuth authentication, which is required by many services (Google, GitHub, Slack).

**Why OAuth is complex:**

*   **Complex Setup:** Requires creating OAuth apps with providers, including redirect URLs, scopes, and credentials.
*   **Implementation Overhead:** Requires handling OAuth 2.0 flows, including callbacks, token refresh, and secure storage.
*   **Credential Management:** Managing OAuth app secrets across various services.
*   **Token Lifecycle:** Managing token expiration, refresh, and error handling.

Klavis AI handles these complexities.

**Alternative:** Advanced users can implement OAuth directly. Refer to individual server READMEs for details.

## 📚 Resources & Community

| Resource              | Link                                                                  | Description                                           |
|-----------------------|-----------------------------------------------------------------------|-------------------------------------------------------|
| **📖 Documentation**    | [docs.klavis.ai](https://docs.klavis.ai)                               | Complete guides and API reference.                    |
| **💬 Discord**        | [Join Community](https://discord.gg/p7TuTEcssn)                        | Get help and connect with other users.                |
| **🐛 Issues**          | [GitHub Issues](https://github.com/klavis-ai/klavis/issues)           | Report bugs and request features.                   |
| **📦 Examples**        | [examples/](examples/)                                              | Working examples for popular AI frameworks.           |
| **🔧 Server Guides**  | [mcp_servers/](mcp_servers/)                                          | Individual server documentation.                      |
| **🚀 Klavis AI Website** | [klavis.ai](https://www.klavis.ai) | Learn more and create an account. |

## 🤝 Contributing

We welcome contributions!  Help improve Klavis AI!

*   Bug Reports / Feature Requests
*   Documentation Improvements
*   New MCP Server Development
*   Existing Server Enhancements

Refer to our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p><strong>🚀 Supercharge Your AI Applications with Klavis AI </strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">View on GitHub</a>
  </p>
</div>
```
Key improvements and SEO considerations:

*   **Concise Hook:** Starts with a clear, benefit-driven opening sentence.
*   **Targeted Keywords:**  Uses relevant keywords like "MCP servers," "AI applications," "self-hosting," "managed service," "OAuth," and specific service names (GitHub, Gmail, etc.) throughout the content.
*   **Clear Headings:** Organizes information logically with descriptive headings and subheadings.
*   **Bullet Points:** Makes key features and benefits easy to scan.
*   **Strong Call to Actions:**  Encourages users to get started with clear calls to action (Get Free API Key, Documentation, etc.).
*   **Detailed Explanations:** Provides enough context for developers to understand the value proposition and how to use the platform.
*   **Code Examples:**  Includes practical code examples for both self-hosting and the hosted service.
*   **Resource Links:**  Includes links to the documentation, Discord, GitHub, and examples to help users find more information.
*   **SEO Optimization:**  Keywords are naturally integrated within headings, subheadings, and body text.  The structure uses common SEO best practices.
*   **Concise and Clear:** The writing is direct and easy to understand.
*   **Internal and External Links:** Added a link back to the original repo and links to external resources.
*   **Emphasis on Benefits:** Highlights the value Klavis AI provides to developers.
*   **Mobile-Friendly:** The markdown format is easily readable on mobile devices.
*   **Updated and Completed Information:** Added API examples, full list of features, and complete example of setup.