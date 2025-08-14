<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Powering AI Applications with Seamless Integrations</h1>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/klavis.svg)](https://pypi.org/project/klavis/)
[![npm](https://img.shields.io/npm/v/klavis.svg)](https://www.npmjs.com/package/klavis)

</div>

**Klavis AI provides open-source MCP (Message Channel Protocol) integrations, enabling rapid development of AI applications by connecting to a wide range of services.**  

## Key Features:

*   **🚀 Instant Integration:** Get up and running in minutes with Python, TypeScript SDKs, or our REST API.
*   **🔐 Secure Authentication:** Built-in OAuth flows and API key management for secure access.
*   **⚡ Production-Ready:** Utilize our hosted infrastructure, ready to scale for millions of users.
*   **🛠️ Extensive Toolset:** Access 100+ pre-built integrations with CRMs, GSuite, GitHub, Slack, databases, and more.
*   **🌐 Broad Compatibility:** Works seamlessly with all major LLM providers (OpenAI, Anthropic, Gemini, etc.) and AI agent frameworks (LangChain, LlamaIndex, CrewAI, AutoGen, etc.).
*   **🔧 Self-Hostable:** Deploy and manage your own open-source MCP servers for complete control.

## Getting Started

### Installation

**Python**

```bash
pip install klavis
```

**TypeScript/JavaScript**

```bash
npm install klavis
```

### Obtain Your API Key

Sign up at [klavis.ai](https://www.klavis.ai) to get your [API key](https://www.klavis.ai/home/api-keys).

## Code Examples

These examples show how to use Klavis with and without an existing MCP client.

### With MCP Client

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

### Without MCP Client (Function Calling)

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

## Tutorials

*   [AI Platform Integrations Overview](https://docs.klavis.ai/documentation/integrations/overview)
*   [Together AI Integration](https://docs.klavis.ai/documentation/integrations/together-ai)
*   [OpenAI Integration](https://docs.klavis.ai/documentation/integrations/open-ai)
*   [View All Servers →](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)

## Roadmap

*   Expanding our library of high-quality MCP Servers (Figma, Canva, Perplexityai, Microsoft teams, Google maps, Bitbucket, Cloudflare, Zoho, Tavily, Posthog, Mem0, Brave Search, Apollo, Exa, Fireflies, Eleven Labs, Hacker News, MicroSoft Calendar)
*   More AI platform integrations ([examples](https://github.com/Klavis-AI/klavis/tree/main/examples) & [docs](https://docs.klavis.ai/documentation/ai-platform-integration/overview) )
*   Event-driven / Webhook functionality.
*   Comprehensive Unit and Integration Tests.
*   Continuous /docs improvement

## Authentication and Multi-Tool Workflows

### Authentication

Klavis simplifies authentication for services requiring it:

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

Take control by self-hosting our open-source MCP servers:

```bash
# Clone the repository
git clone https://github.com/klavis-ai/klavis.git
cd klavis

# Run a specific MCP server
cd mcp_servers/github
docker build -t klavis-github .
docker run -p 8000:8000 klavis-github
```
*checkout each readme for more details*

## Documentation

*   [API Documentation](https://docs.klavis.ai)
*   [SDK Documentation](https://docs.klavis.ai/sdks)
*   [MCP Protocol Guide](https://docs.klavis.ai/mcp)
*   [Authentication Guide](https://docs.klavis.ai/auth)

## Contributing

We encourage contributions!

1.  **Report Issues**: Found a bug? [Open an issue](https://github.com/Klavis-AI/klavis/issues)
2.  **Suggest Features**: Have an idea? [Start a discussion](https://github.com/Klavis-AI/klavis/discussions)
3.  **Contribute Code**: Review our [Contributing Guidelines](CONTRIBUTING.md)
4.  **Build MCP Servers**:  See our [MCP Server Guide](MCP_SERVER_GUIDE.md)
5.  **Join Community**: Connect with us on [Discord](https://discord.gg/p7TuTEcssn)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Ready to accelerate your AI development?</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Started</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a> •
    <a href="https://github.com/Klavis-AI/klavis">GitHub Repository</a>
  </p>
</div>
```

Key improvements and SEO considerations:

*   **Strong Opening Hook:** Immediately states the value proposition.
*   **Clear Headings:** Uses H1, H2, and H3 for structure.
*   **Bulleted Key Features:**  Highlights benefits using concise bullet points, important for readability and SEO.
*   **Keyword Optimization:** Includes relevant keywords like "AI integrations," "open source," "MCP," and service names.
*   **Call to Action:** Multiple CTAs throughout, including a clear call to action at the end.
*   **Internal Links:** Links to documentation, examples, and the GitHub repo to encourage exploration.
*   **Concise Language:**  Removes unnecessary wording.
*   **GitHub Repository Link:** Added a direct link to the GitHub repo at the end.
*   **Alt Text:** Included `alt` text for the logo image for accessibility and SEO.