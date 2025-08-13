<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Open-Source MCP Integrations for Seamless AI Application Development</h1>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/klavis.svg)](https://pypi.org/project/klavis/)
[![npm](https://img.shields.io/npm/v/klavis.svg)](https://www.npmjs.com/package/klavis)

</div>

## 💡 Supercharge Your AI Applications with Klavis AI

**Klavis AI provides open-source MCP (Message Channel Protocol) integrations, empowering developers to build advanced AI applications with ease.** Eliminate the complexities of authentication and client-side code, and integrate with numerous platforms using our hosted, high-quality, and secure MCP servers.

## ✨ Key Features for Rapid AI Integration

*   **🚀 Instant Integration:** Get started in minutes with our Python and TypeScript SDKs, or leverage our straightforward REST API.
*   **🔐 Secure Authentication:** Benefit from built-in OAuth flows and API key management for secure connections.
*   **⚡ Production-Ready Infrastructure:** Rely on our hosted infrastructure that scales to support millions of users.
*   **🛠️ Extensive Tool Library:** Access over 100 tools, including integrations for CRM, GSuite, GitHub, Slack, databases, and more.
*   **🌐 Cross-Platform Compatibility:** Seamlessly integrate with any LLM provider (OpenAI, Anthropic, Gemini, etc.) and AI agent framework (LangChain, Llamaindex, CrewAI, AutoGen, etc.).
*   **🔧 Self-Hosting Capabilities:** Run open-source MCP servers on your own infrastructure for enhanced control and privacy.

## 🚀 Quick Start: Integrate Klavis AI in Minutes

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

Sign up at [klavis.ai](https://www.klavis.ai) and create your [API key](https://www.klavis.ai/home/api-keys).

## 💻 Integration Examples

### Integrating with an Existing MCP Client

**Python Example**

```python
from klavis import Klavis
from klavis.types import McpServerName, ConnectionType

klavis_client = Klavis(api_key="your-klavis-key")

# Create a YouTube MCP server instance
youtube_server = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.YOUTUBE,
    user_id="user123",  # Replace with your user ID
    platform_name="MyApp"  # Replace with your platform name
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

### Integrating Without an MCP Client (Function Calling)

Directly integrate with your preferred LLM provider or AI agent framework by utilizing function calling:

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

## 📚 Comprehensive AI Platform Integration Tutorials

*   **[AI Platform Integrations Overview](https://docs.klavis.ai/documentation/integrations/overview)** - Learn how to seamlessly integrate with leading AI platforms.
*   **[Together AI Integration](https://docs.klavis.ai/documentation/integrations/together-ai)** - Build AI agents with Together AI's high-performance LLMs.
*   **[OpenAI Integration](https://docs.klavis.ai/documentation/integrations/open-ai)** - Create fast and efficient AI agents using OpenAI and Klavis MCP Servers.
*   **And More!** Explore a growing library of integration guides.

## 🛠️ Available MCP Servers

[**View All Servers →**](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)

## 🗺️ Roadmap: Expanding Capabilities

*   **New MCP Servers:**  Figma, Canva, PerplexityAI, Microsoft Teams, Google Maps, Bitbucket, Cloudflare, Zoho, Tavily, Posthog, Mem0, Brave Search, Apollo, Exa, Fireflies, Eleven Labs, Hacker News, Microsoft Calendar, and more.
*   **Expanded AI Platform Integrations:** Continuously adding integrations to support leading platforms. [See Examples](https://github.com/Klavis-AI/klavis/tree/main/examples) and [Documentation](https://docs.klavis.ai/documentation/ai-platform-integration/overview).
*   **Event-Driven & Webhook Support:** Enable real-time interactions and automation.
*   **Robust Testing:** Implement comprehensive unit and integration tests.
*   **Documentation Enhancements:** Continuously improving our documentation for clarity and usability.

## 🔧 Streamlined Authentication & Multi-Tool Workflows

### Authentication

Klavis AI simplifies authentication, essential for many MCP servers:

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

## 🏠 Self-Hosting for Maximum Control

Take full control by self-hosting our open-source MCP servers:

```bash
# Clone the repository
git clone https://github.com/klavis-ai/klavis.git
cd klavis

# Run a specific MCP server
cd mcp_servers/github
docker build -t klavis-github .
docker run -p 8000:8000 klavis-github
```

(Check each server's `README` for specific instructions.)

## 📖 Essential Documentation Resources

*   **[API Documentation](https://docs.klavis.ai)** - Your complete API reference.
*   **[SDK Documentation](https://docs.klavis.ai/sdks)** - Python and TypeScript guides for quick integration.
*   **[MCP Protocol Guide](https://docs.klavis.ai/mcp)** - Deep dive into the MCP protocol.
*   **[Authentication Guide](https://docs.klavis.ai/auth)** - Understand OAuth and API keys.

## 🤝 Join the Klavis AI Community: Contribute and Collaborate

We welcome contributions from the community!

1.  **Report Issues:** Found a bug? [Open an issue](https://github.com/klavis-ai/klavis/issues) and let us know.
2.  **Request Features:** Have an idea for a new integration or improvement? [Start a discussion](https://github.com/klavis-ai/klavis/discussions).
3.  **Contribute Code:**  Refer to our [Contributing Guidelines](CONTRIBUTING.md) to learn how to contribute.
4.  **Build MCP Servers:**  Want to add a new integration?  See our [MCP Server Guide](MCP_SERVER_GUIDE.md).
5.  **Engage with the Community:**  Connect with us and other developers on [Discord](https://discord.gg/p7TuTEcssn).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Ready to revolutionize your AI applications?</strong></p>
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

*   **Hook:** Added a compelling one-sentence hook at the start.
*   **Headings:**  Used clear, keyword-rich headings to improve readability and SEO (e.g., "Quick Start," "Key Features").
*   **Keywords:** Incorporated relevant keywords naturally throughout the text (e.g., "AI applications," "open source," "MCP integrations," "LLM," "AI agent").
*   **Bulleted Lists:**  Used bulleted lists to emphasize key features and benefits, improving readability.
*   **Action-Oriented Language:** Used strong verbs and calls to action (e.g., "Supercharge," "Get Started," "Explore").
*   **Internal Links:** Included links to the documentation, examples, and other important sections within the README.
*   **External Link to GitHub:** Included a link back to the original repository, as requested.
*   **Concise Language:**  Streamlined the text to make it more direct and to the point.
*   **Formatting:** Maintained the markdown formatting for good readability.
*   **Alt Text:** Added alt text to the logo image for accessibility.
*   **SEO-Friendly Title:**  The overall title is concise and descriptive, incorporating relevant keywords.
*   **Clear Code Examples:** The quickstart examples are well-formatted and easy to understand.
*   **Complete:** Includes all sections of the original README and additional sections with improvements.