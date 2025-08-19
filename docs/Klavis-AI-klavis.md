<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Seamlessly Integrate AI Applications with Open Source MCP Integrations</h1>

<div align="center">
  <a href="https://docs.klavis.ai">
    <img src="https://img.shields.io/badge/Documentation-📖-green" alt="Documentation">
  </a>
  <a href="https://www.klavis.ai">
    <img src="https://img.shields.io/badge/Website-🌐-purple" alt="Website">
  </a>
  <a href="https://discord.gg/p7TuTEcssn">
    <img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://pypi.org/project/klavis/">
    <img src="https://img.shields.io/pypi/v/klavis.svg" alt="PyPI">
  </a>
  <a href="https://www.npmjs.com/package/klavis">
    <img src="https://img.shields.io/npm/v/klavis.svg" alt="npm">
  </a>
</div>

**Klavis AI provides open-source MCP integrations to effortlessly connect your AI applications to various services, streamlining development and enhancing functionality.**  [View the Klavis AI Repository](https://github.com/Klavis-AI/klavis)

## Key Features of Klavis AI

*   **🚀 Instant Integration:** Quickly integrate with Python and TypeScript SDKs or REST API in minutes.
*   **🔐 Built-in Authentication:** Secure your integrations with pre-built OAuth flows and API key management.
*   **⚡ Production-Ready:** Leverage a hosted infrastructure designed to scale for millions of users.
*   **🛠️ Extensive Tool Library:** Access a wide range of tools, including CRM, GSuite, GitHub, Slack, and databases.
*   **🌐 Universal Compatibility:** Works with any LLM provider (OpenAI, Anthropic, Gemini, etc.) and AI agent framework (LangChain, Llamaindex, CrewAI, AutoGen, etc.).
*   **🔧 Self-Hosting Capability:** Run open-source MCP servers on your own infrastructure for complete control.

## Quick Start Guide

### Installation

**Python:**

```bash
pip install klavis
```

**TypeScript/JavaScript:**

```bash
npm install klavis
```

#### Get Your API Key

1.  Sign up at [klavis.ai](https://www.klavis.ai)
2.  Create your [API key](https://www.klavis.ai/home/api-keys).

## Code Examples

### With MCP Client

If you already have an MCP client implementation:

**Python Example**

```python
from klavis import Klavis
from klavis.types import McpServerName

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

### Without MCP Client (Function Calling)

Integrate directly with your LLM or AI agent framework using function calling:

**Python + OpenAI Example**

```python
import json
from openai import OpenAI
from klavis import Klavis
from klavis.types import McpServerName, ToolFormat

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

## AI Platform Integration Tutorials

*   **[AI Platform Integrations Overview](https://docs.klavis.ai/documentation/integrations/overview)** - Learn how to integrate with leading AI platforms
*   **[Together AI Integration](https://docs.klavis.ai/documentation/integrations/together-ai)** - Build AI agents with Together AI's high-performance LLMs
*   **[OpenAI Integration](https://docs.klavis.ai/documentation/integrations/open-ai)** - Create fast and efficient AI agents with OpenAI and Klavis MCP Servers
*   And More!

## Available MCP Servers

[**View All Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)

## Roadmap

*   More high-quality MCP Servers (Figma, Canva, Perplexityai, Microsoft Teams, Google Maps, Bitbucket, Cloudflare, Zoho, Tavily, Posthog, Mem0, Brave Search, Apollo, Exa, Fireflies, Eleven Labs, Hacker News, Microsoft Calendar)
*   More AI platform integrations ([examples](https://github.com/Klavis-AI/klavis/tree/main/examples) & [docs](https://docs.klavis.ai/documentation/ai-platform-integration/overview) )
*   Event-driven / Webhook Support
*   Unit and Integration Testing
*   Documentation Improvements

## Authentication and Multi-Tool Workflows

### Authentication

Klavis simplifies authentication, handling both OAuth and API key management:

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

## Self-Hosting

All Klavis AI servers are open-source, allowing you to run them yourself.

```bash
# Clone the repository
git clone https://github.com/klavis-ai/klavis.git
cd klavis

# Run a specific MCP server
cd mcp_servers/github
docker build -t klavis-github .
docker run -p 8000:8000 klavis-github
```
checkout each readme for more details

## Documentation

*   **[API Documentation](https://docs.klavis.ai)** - Comprehensive API reference
*   **[SDK Documentation](https://docs.klavis.ai/sdks)** - Guides for Python & TypeScript
*   **[MCP Protocol Guide](https://docs.klavis.ai/mcp)** - Understanding the MCP protocol
*   **[Authentication Guide](https://docs.klavis.ai/auth)** - Learn about OAuth and API key usage

## Contributing

We welcome contributions!

1.  **Report Issues**: [Open an issue](https://github.com/klavis-ai/klavis/issues) if you find a bug.
2.  **Request Features**: [Start a discussion](https://github.com/klavis-ai/klavis/discussions) for new ideas.
3.  **Contribute Code**: Review our [Contributing Guidelines](CONTRIBUTING.md) to get started.
4.  **Build MCP Servers**: See our [MCP Server Guide](MCP_SERVER_GUIDE.md) to add new integrations.
5.  **Join Community**: Connect with us on [Discord](https://discord.gg/p7TuTEcssn).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Ready to supercharge your AI applications?</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Started</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>