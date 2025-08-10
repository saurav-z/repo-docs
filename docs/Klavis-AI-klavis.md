<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Integrate Any App, Power Any AI</h1>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/klavis.svg)](https://pypi.org/project/klavis/)
[![npm](https://img.shields.io/npm/v/klavis.svg)](https://www.npmjs.com/package/klavis)

</div>

## What is Klavis AI?

**Klavis AI is an open-source platform that simplifies the integration of AI applications with a wide range of services, eliminating the complexities of authentication and client-side code.** Offering hosted, secure, and high-quality MCP (Machine Capability Protocol) integrations, Klavis empowers developers to quickly connect their AI agents to tools like CRM, GSuite, databases, and more, enhancing functionality and user experience.

## ✨ Key Features:

*   **🚀 Instant Integration:** Integrate in minutes with Python and TypeScript SDKs, or through our REST API.
*   **🔐 Secure Authentication:** Built-in OAuth flows and API key management for hassle-free security.
*   **⚡ Production-Ready:** Scalable hosted infrastructure for handling millions of users.
*   **🛠️ Extensive Tool Library:** Access to 100+ integrations, including CRM, GSuite, GitHub, Slack, and databases.
*   **🌐 Broad Compatibility:** Works seamlessly with any LLM provider (OpenAI, Anthropic, Gemini, etc.) and AI agent framework (LangChain, Llamaindex, CrewAI, AutoGen, etc.).
*   **🔧 Self-Hosting Options:** Open-source MCP servers for you to host and control.

## 🚀 Quick Start

### Installation

**Python**

```bash
pip install klavis
```

**TypeScript/JavaScript**

```bash
npm install klavis
```

#### Get Your API Key

Sign up at [klavis.ai](https://www.klavis.ai) to obtain your API key, crucial for accessing Klavis AI services.  Create your [API key](https://www.klavis.ai/home/api-keys).

## Integrating with an MCP Client

If you already use an MCP client in your codebase:

**Python Example**

```python
from klavis import Klavis
from klavis.types import McpServerName, ConnectionType

klavis_client = Klavis(api_key="your-klavis-key")

# Create a YouTube MCP server instance
youtube_server = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.YOUTUBE,
    user_id="user123", # Change to user id in your platform
    platform_name="MyApp" # change to your platform
)

print(f"Server created: {youtube_server.server_url}")
```

**TypeScript Example**

```typescript
import { KlavisClient, Klavis } from 'klavis';

const klavisClient = new KlavisClient({ apiKey: 'your-klavis-key' });

// Create Gmail MCP server with OAuth
const gmailServer = await klavisClient.mcpServer.createServerInstance({
    serverName: Klavis.McpServerName.Gmail,
    userId: "user123",
    platformName: "MyApp"
});

// Gmail needs OAuth flow
await window.open(gmailServer.oauthUrl);
```

## Integrating Without an MCP Client (Function Calling)

Directly integrate with your LLM provider or AI agent framework via function calling:

**Python + OpenAI Example**

```python
import json
from openai import OpenAI
from klavis import Klavis
from klavis.types import McpServerName, ConnectionType, ToolFormat

OPENAI_MODEL = "gpt-4o-mini"

openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
klavis_client = Klavis(api_key="YOUR_KLAVIS_API_KEY")

# Create server instance
youtube_server = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.YOUTUBE,
    user_id="user123",
    platform_name="MyApp"
)

# Get available tools in OpenAI format
tools = klavis_client.mcp_server.list_tools(
    server_url=youtube_server.server_url,
    format=ToolFormat.OPENAI,
)

# Initial conversation
messages = [{"role": "user", "content": "Summarize this video: https://youtube.com/watch?v=..."}]

# First OpenAI call with function calling
response = openai_client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=messages,
    tools=tools.tools
)

messages.append(response.choices[0].message)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        result = klavis_client.mcp_server.call_tools(
            server_url=youtube_server.server_url,
            tool_name=tool_call.function.name,
            tool_args=json.loads(tool_call.function.arguments),
        )
        
        # Add tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

# Second OpenAI call to process tool results and generate final response
final_response = openai_client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=messages
)

print(final_response.choices[0].message.content)
```

**TypeScript + OpenAI Example**

```typescript
import OpenAI from 'openai';
import { KlavisClient, Klavis } from 'klavis';

// Constants
const OPENAI_MODEL = "gpt-4o-mini";

const EMAIL_RECIPIENT = "john@example.com";
const EMAIL_SUBJECT = "Hello from Klavis";
const EMAIL_BODY = "This email was sent using Klavis MCP Server!";

const openaiClient = new OpenAI({ apiKey: 'your-openai-key' });
const klavisClient = new KlavisClient({ apiKey: 'your-klavis-key' });

// Create server and get tools
const gmailServer = await klavisClient.mcpServer.createServerInstance({
    serverName: Klavis.McpServerName.Gmail,
    userId: "user123",
    platformName: "MyApp"
});

// Handle OAuth authentication for Gmail
if (gmailServer.oauthUrl) {
    console.log("Please complete OAuth authorization:", gmailServer.oauthUrl);
    await window.open(gmailServer.oauthUrl);
}

const tools = await klavisClient.mcpServer.listTools({
    serverUrl: gmailServer.serverUrl,
    format: Klavis.ToolFormat.Openai
});

// Initial conversation
const messages = [{ 
    role: "user", 
    content: `Please send an email to ${EMAIL_RECIPIENT} with subject "${EMAIL_SUBJECT}" and body "${EMAIL_BODY}"` 
}];

// First OpenAI call with function calling
const response = await openaiClient.chat.completions.create({
    model: OPENAI_MODEL,
    messages: messages,
    tools: tools.tools
});

messages.push(response.choices[0].message);

// Handle tool calls
if (response.choices[0].message.tool_calls) {
    for (const toolCall of response.choices[0].message.tool_calls) {
        const result = await klavisClient.mcpServer.callTools({
            serverUrl: gmailServer.serverUrl,
            toolName: toolCall.function.name,
            toolArgs: JSON.parse(toolCall.function.arguments)
        });
        
        // Add tool result to conversation
        messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify(result)
        });
    }
}

// Second OpenAI call to process tool results and generate final response
const finalResponse = await openaiClient.chat.completions.create({
    model: OPENAI_MODEL,
    messages: messages
});

console.log(finalResponse.choices[0].message.content);
```

## 📚 AI Platform Integration Tutorials

*   **[AI Platform Integrations Overview](https://docs.klavis.ai/documentation/integrations/overview)** - Learn how to integrate with leading AI platforms
*   **[Together AI Integration](https://docs.klavis.ai/documentation/integrations/together-ai)** - Build AI agents with Together AI's high-performance LLMs
*   **[OpenAI Integration](https://docs.klavis.ai/documentation/integrations/open-ai)** - Create fast and efficient AI agents with OpenAI and Klavis MCP Servers
*   And More!

## 🛠️ Available MCP Servers

[**View All Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)

## 🗺️ Roadmap

*   Add support for more high quality MCP Servers (Figma, Canva, Perplexityai, Microsoft teams, Google maps, Bitbucket, Cloudflare, Zoho, Tavily, Posthog, Mem0, Brave Search, Apollo, Exa, Fireflies, Eleven Labs, Hacker News, MicroSoft Calendar).
*   More AI platform integrations ([examples](https://github.com/Klavis-AI/klavis/tree/main/examples) & [docs](https://docs.klavis.ai/documentation/ai-platform-integration/overview) )
*   Event-driven / Webhook support.
*   Unit and integration tests.
*   Improvement of the `/docs`.

## 🔧 Authentication & Multi-Tool Workflows

### Authentication

Klavis streamlines the authentication process for services requiring it:

```python
# For OAuth services (Gmail, Google Drive, etc.)
server = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.GMAIL,
    user_id="user123",
    platform_name="MyApp"
)
# Option 1 - OAuth URL is provided in server.oauth_url, redirect user to OAuth URL for authentication
import webbrowser
webbrowser.open(server.oauth_url)

# Option 2 - or for API key services
klavis_client.mcp_server.set_auth_token(
    instance_id=server.instance_id,
    auth_token="your-service-api-key"
)
```

## 🏠 Self-Hosting

Run MCP servers on your own infrastructure with Klavis' open-source servers:

```bash
# Clone the repository
git clone https://github.com/klavis-ai/klavis.git
cd klavis

# Run a specific MCP server
cd mcp_servers/github
docker build -t klavis-github .
docker run -p 8000:8000 klavis-github
```
See each `README` file for detailed instructions.

## 📖 Documentation

*   **[API Documentation](https://docs.klavis.ai)** - Complete API reference
*   **[SDK Documentation](https://docs.klavis.ai/sdks)** - Python & TypeScript guides
*   **[MCP Protocol Guide](https://docs.klavis.ai/mcp)** - Understanding MCP
*   **[Authentication Guide](https://docs.klavis.ai/auth)** - OAuth and API keys

## 🤝 Contributing

We welcome contributions! Get involved by:

1.  **Reporting Issues**: Found a bug? [Open an issue](https://github.com/klavis-ai/klavis/issues)
2.  **Requesting Features**: Have an idea? [Start a discussion](https://github.com/klavis-ai/klavis/discussions)
3.  **Contributing Code**: Review our [Contributing Guidelines](CONTRIBUTING.md)
4.  **Building MCP Servers**: Eager to add integrations? Check out our [MCP Server Guide](MCP_SERVER_GUIDE.md)
5.  **Join the Community**: Connect with us on [Discord](https://discord.gg/p7TuTEcssn)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Ready to supercharge your AI applications?</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Started</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
    <br>
    <a href="https://github.com/Klavis-AI/klavis">View on GitHub</a>
  </p>
</div>
```
Key improvements and SEO considerations:

*   **Hook:**  The one-sentence hook at the beginning is compelling and includes a crucial keyword: "Klavis AI: Integrate Any App, Power Any AI."
*   **Headings:**  Clear, descriptive headings are used throughout for better readability and SEO.
*   **Keywords:** Strategic use of keywords like "AI integration," "open source," "MCP," "API," and names of popular tools.
*   **Bulleted Lists:**  Features are presented in a bulleted list for easy skimming and emphasis.
*   **Concise Descriptions:**  Each section is written clearly and to the point.
*   **Call to Actions:**  Encourage the reader to explore the docs, get started, or join the community.
*   **GitHub Link:** The link back to the original repo is included.
*   **Alt Text:** Added `alt` text to the logo for accessibility and SEO.
*   **Structure:** Improved formatting makes the information easier to find.