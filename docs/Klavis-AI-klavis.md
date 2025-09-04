<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Your Gateway to Production-Ready MCP Servers</h1>

<p align="center"><b>Easily integrate with popular services and tools with our self-hosted and hosted solutions.</b></p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker Images](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://github.com/orgs/klavis-ai/packages)

</div>

---

## 🚀 Key Features of Klavis AI

Klavis AI simplifies the integration of your applications with various services, providing both self-hosted and hosted options.

*   ✅ **Production-Ready MCP Servers:**  Access a wide range of pre-built MCP (Metadata Control Protocol) servers, ready for immediate use.
*   ✅ **Self-Hosting with Docker:** Easily deploy MCP servers using Docker, offering flexibility and control over your infrastructure.
*   ✅ **Hosted MCP Service:**  Leverage our managed infrastructure for a hassle-free, production-ready experience with automatic scaling and updates.
*   ✅ **Enterprise-Grade Security:** Benefit from robust security measures, including Enterprise OAuth support and SOC2 compliance.
*   ✅ **Extensive Integrations:** Seamlessly integrate with 50+ services, including Gmail, GitHub, Slack, and more.
*   ✅ **Rapid Deployment:** Deploy MCP servers within seconds, both with Docker and our hosted service.

---

## 📦 Self-Hosting Klavis AI MCP Servers

Deploy Klavis AI MCP servers on your own infrastructure using Docker or by building from source code. This section explains how to get started.

### 🐳 Docker Deployment (Recommended)

Quickly set up MCP servers using Docker images.

1.  **Pull the Docker Image:**

    ```bash
    docker pull ghcr.io/klavis-ai/github-mcp-server:latest
    ```

2.  **Run the Server (with OAuth):**  Requires a Klavis API key.

    ```bash
    docker run -p 5000:5000 -e KLAVIS_API_KEY=$KLAVIS_API_KEY ghcr.io/klavis-ai/github-mcp-server:latest
    ```

3.  **Run the Server (without OAuth):**

    ```bash
    docker run -p 5000:5000 -e AUTH_DATA='{"access_token":"ghp_your_github_token_here"}' ghcr.io/klavis-ai/github-mcp-server:latest
    ```

    *   The MCP server typically runs on port 5000 and exposes the MCP protocol at the `/mcp` path.
    *   **Browse available images: [Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)**

### 🏗️ Build from Source

Alternatively, build and run MCP servers from source code.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Klavis-AI/klavis.git
    cd klavis/mcp_servers/github
    ```

2.  **Choose your build method:**

    *   **Using Docker:**
        ```bash
        docker build -t github-mcp .
        docker run -p 5000:5000 github-mcp
        ```
    *   **Run directly (Go example):**
        ```bash
        go mod download
        go run server.go
        ```
    *   **Run Python servers:**
        ```bash
        cd ../youtube
        pip install -r requirements.txt
        python server.py
        ```
    *   **Run Node.js servers:**
        ```bash
        cd ../slack
        npm install
        npm start
        ```

---

## 🌐 Hosted MCP Service: The Easiest Path to Integration

Our hosted MCP service provides a hassle-free solution for integrating with various services, eliminating the need for infrastructure management.

### ✨ Benefits of the Hosted Service:

*   **Instant Setup:** Deploy MCP servers in under 30 seconds.
*   **Simplified OAuth:**  Handle OAuth authentication without complex configurations.
*   **Managed Infrastructure:**  Focus on your applications without managing infrastructure.
*   **Automatic Scaling:**  Benefit from automatic scaling to accommodate your needs.
*   **Always Updated:**  Access the latest MCP server versions automatically.
*   **Cost-Effective:**  Pay only for the resources you consume, with a free tier available.

### 💻 Quick Integration with the Hosted Service

1.  **Install the Klavis Python package:**

    ```bash
    pip install klavis  # or npm install klavis
    ```

2.  **Get your API Key:**  [Get Free API Key](https://www.klavis.ai/home/api-keys)

3.  **Use the following code:**

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

4.  **Configure your tools**

    *   🔗 **[Visit our MCP Servers page →](https://www.klavis.ai/home/mcp-servers)**
    *   **Select any service** (Gmail, GitHub, Slack, etc.)
    *   **Copy the generated configuration** for your tool
    *   **Paste into Claude Desktop config** - done!

---

## 🔐 OAuth Authentication

Klavis AI simplifies OAuth implementation for services like Google, GitHub, and Slack. Our hosted service handles the complexities of OAuth, allowing you to focus on your integration.  For self-hosting, an API key is required to leverage Klavis-managed OAuth for supported servers.

---

## 🛠️ Available MCP Servers

Explore the available MCP servers with examples:

| Service         | Docker Image                              | OAuth Required | Description                         |
|-----------------|-------------------------------------------|----------------|-------------------------------------|
| **GitHub**      | `ghcr.io/klavis-ai/github-mcp-server`     | ✅             | Repository management, issues, PRs    |
| **Gmail**       | `ghcr.io/klavis-ai/gmail-mcp-server:latest`   | ✅             | Email reading, sending, management  |
| **Google Sheets** | `ghcr.io/klavis-ai/google_sheets-mcp-server:latest` | ✅             | Spreadsheet operations                |
| **YouTube**     | `ghcr.io/klavis-ai/youtube-mcp-server`    | ❌             | Video information, search           |
| **Slack**       | `ghcr.io/klavis-ai/slack-mcp-server:latest`   | ✅             | Channel management, messaging         |
| **Notion**      | `ghcr.io/klavis-ai/notion-mcp-server:latest`  | ✅             | Database and page operations        |
| **Salesforce**  | `ghcr.io/klavis-ai/salesforce-mcp-server:latest` | ✅             | CRM data management                 |
| **Postgres**    | `ghcr.io/klavis-ai/postgres-mcp-server`   | ❌             | Database operations                 |
| ...             | ...                                       | ...            | ...                                 |

**View More Servers**: [All 50+ Servers](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart) | [Browse Docker Images](https://github.com/orgs/Klavis-AI/packages?repo_name=klavis)

---

## 💡 Usage Examples

Integrate with existing MCP implementations:

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

**📖 More Examples:** [examples/](examples/)

---

## 📚 Resources & Community

Access helpful resources and connect with the Klavis AI community.

| Resource         | Link                                  | Description                                 |
|------------------|---------------------------------------|---------------------------------------------|
| **Documentation** | [docs.klavis.ai](https://docs.klavis.ai) | Complete guides and API reference             |
| **Discord**       | [Join Community](https://discord.gg/p7TuTEcssn) | Get help and connect with other users        |
| **Issues**        | [GitHub Issues](https://github.com/klavis-ai/klavis/issues) | Report bugs and request new features        |
| **Examples**      | [examples/](examples/)                 | Working examples with popular AI frameworks |
| **Server Guides** | [mcp_servers/](mcp_servers/)         | Individual server documentation               |

---

## 🤝 Contributing

Help us improve Klavis AI!  We welcome contributions for bug fixes, documentation, new MCP servers, and enhancements.

*   **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📜 License

Klavis AI is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <p><strong>🚀 Supercharge AI Applications with Klavis AI!</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Free API Key</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>
```
Key improvements and SEO considerations:

*   **Clear Hook:** A strong opening sentence that grabs attention.
*   **SEO Keywords:** Integrated relevant keywords (e.g., "MCP server," "self-hosted," "hosted service," "integrations," "OAuth") naturally.
*   **Structured Content:** Headings, subheadings, and bullet points improve readability and SEO.
*   **Concise Descriptions:**  Clear and brief explanations of features and benefits.
*   **Calls to Action:**  Repeated links to get a free API key, documentation, and other resources.
*   **Alt Text:** Added `alt` text to the logo image for accessibility and SEO.
*   **Keyword Density:** Maintained a good keyword density without being overly repetitive.
*   **Internal Linking:**  Links within the README to other sections and resources, improving navigation and SEO.
*   **Target Audience:** Appeals to both developers and those who want an easy solution.
*   **Markdown Formatting:**  Uses correct Markdown for proper rendering on GitHub and other platforms.
*   **Clear Instructions:** Provides easy-to-follow instructions for setup and usage.
*   **Organized Information:** Improves clarity and user experience.
*   **Focus on Value:**  Highlights the key benefits of Klavis AI to attract users.
*   **Concise language**: Avoids unnecessary wordiness.