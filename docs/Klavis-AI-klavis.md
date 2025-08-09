<div align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/klavis-ai/klavis/main/static/klavis-ai.png" width="80" alt="Klavis AI Logo">
  </picture>
</div>

<h1 align="center">Klavis AI: Open-Source MCP Integrations for AI Applications</h1>

<div align="center">

[![Documentation](https://img.shields.io/badge/Documentation-📖-green)](https://docs.klavis.ai)
[![Website](https://img.shields.io/badge/Website-🌐-purple)](https://www.klavis.ai)
[![Discord](https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white)](https://discord.gg/p7TuTEcssn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/klavis.svg)](https://pypi.org/project/klavis/)
[![npm](https://img.shields.io/npm/v/klavis.svg)](https://www.npmjs.com/package/klavis)

</div>

**Klavis AI empowers you to quickly and securely integrate AI applications with various services using open-source MCP integrations.**

## Key Features of Klavis AI

*   🚀 **Instant Integration:** Get up and running in minutes with Python and TypeScript SDKs, or via REST API.
*   🔐 **Built-in Authentication:** Seamlessly handle secure OAuth flows and API key management.
*   ⚡ **Production-Ready:** Utilize hosted infrastructure that scales to support millions of users.
*   🛠️ **Extensive Toolset:** Access a library of 100+ integrations for CRMs, GSuite, GitHub, Slack, databases, and more.
*   🌐 **Versatile Compatibility:** Works with any LLM provider (OpenAI, Anthropic, Gemini, etc.) and any AI agent framework (LangChain, Llamaindex, CrewAI, AutoGen, etc.).
*   🔧 **Self-Hostable:** Deploy and run your own open-source MCP servers.

## Getting Started: Quick Installation

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

1.  Sign up at [klavis.ai](https://www.klavis.ai) to create your account.
2.  Generate your [API key](https://www.klavis.ai/home/api-keys).

## Example Integrations

### With MCP Client

If your codebase already uses an MCP client:

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

Integrate directly with your LLM provider or AI agent framework:

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

## Explore Further: Tutorials and Guides

*   **[AI Platform Integrations Overview](https://docs.klavis.ai/documentation/integrations/overview)**: Learn how to integrate with popular AI platforms.
*   **[Together AI Integration](https://docs.klavis.ai/documentation/integrations/together-ai)**: Build AI agents with Together AI's high-performance LLMs.
*   **[OpenAI Integration](https://docs.klavis.ai/documentation/integrations/open-ai)**: Create fast and efficient AI agents using OpenAI and Klavis MCP Servers.
*   **[View All Servers →](https://docs.klavis.ai/documentation/introduction#mcp-server-quickstart)**: Explore the full range of available servers.

## Roadmap

*   🚀 Expanding MCP Server Library: Plans to add support for Figma, Canva, Perplexity AI, Microsoft Teams, Google Maps, Bitbucket, Cloudflare, Zoho, Tavily, Posthog, Mem0, Brave Search, Apollo, Exa, Fireflies, Eleven Labs, Hacker News, Microsoft Calendar, and more.
*   🤖 Enhanced AI Platform Integrations: More examples and comprehensive documentation ([examples](https://github.com/Klavis-AI/klavis/tree/main/examples) & [docs](https://docs.klavis.ai/documentation/ai-platform-integration/overview) )
*   🔄 Advanced Features: Explore event-driven functionalities and webhooks.
*   ✅ Rigorous Testing: Implementation of unit and integration tests.
*   📚 Documentation Refinement: Continuous improvements to the /docs.

## Authentication and Multi-Tool Workflows

### Authentication

Klavis AI simplifies authentication:

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

## Self-Hosting Klavis AI

Take control by self-hosting the open-source MCP servers:

```bash
# Clone the repository
git clone https://github.com/klavis-ai/klavis.git
cd klavis

# Run a specific MCP server
cd mcp_servers/github
docker build -t klavis-github .
docker run -p 8000:8000 klavis-github
```
Refer to each server's README for detailed instructions.

## Comprehensive Documentation

*   **[API Documentation](https://docs.klavis.ai)**: Detailed API reference.
*   **[SDK Documentation](https://docs.klavis.ai/sdks)**: Guides for Python and TypeScript SDKs.
*   **[MCP Protocol Guide](https://docs.klavis.ai/mcp)**: Deep dive into the MCP protocol.
*   **[Authentication Guide](https://docs.klavis.ai/auth)**: Understand OAuth and API key management.

## Contributing

We welcome your contributions!

1.  **Report Issues**: Found a bug? [Open an issue](https://github.com/klavis-ai/klavis/issues).
2.  **Suggest Features**: Have an idea? [Start a discussion](https://github.com/klavis-ai/klavis/discussions).
3.  **Contribute Code**: See our [Contributing Guidelines](CONTRIBUTING.md).
4.  **Build MCP Servers**: Add new integrations - see the [MCP Server Guide](MCP_SERVER_GUIDE.md).
5.  **Join the Community**: Connect with us on [Discord](https://discord.gg/p7TuTEcssn).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Ready to Supercharge Your AI Applications?</strong></p>
  <p>
    <a href="https://www.klavis.ai">Get Started</a> •
    <a href="https://docs.klavis.ai">Documentation</a> •
    <a href="https://discord.gg/p7TuTEcssn">Discord</a> •
    <a href="examples/">Examples</a>
  </p>
</div>

</a>
```

Key improvements and SEO optimizations:

*   **Clear Title:** The title now includes the core keyword: "Klavis AI" and "Open-Source MCP Integrations".
*   **Concise Hook:** The first sentence immediately highlights the core value proposition: "Klavis AI empowers you to quickly and securely integrate AI applications with various services using open-source MCP integrations."
*   **Keyword Optimization:**  The README uses relevant keywords throughout, like "AI applications," "MCP," "integrations," "open-source," "SDK," "API," and names of integrations.
*   **Structured Headings:** Uses proper H1, H2, and H3 headings for better readability and SEO.
*   **Bulleted Key Features:** Uses bullet points to clearly present the main advantages.
*   **Code Examples:** The code examples are still present and are well-formatted, helping users get started.
*   **Clear Calls to Action:** Includes clear instructions for installation, obtaining an API key, and getting started.
*   **Internal Linking:** Links to specific sections of the documentation are present to encourage user exploration.
*   **Complete Information:**  Includes installation, examples, roadmap, contributing guidelines, and licensing information to ensure users have all the information they need.
*   **Emphasis on Self-Hosting:** Highlights the self-hosting capabilities to attract users who prioritize control and privacy.
*   **Consistent Formatting:** Formatting is consistent throughout for better readability.
*   **Removed redundant badges:** The previous README was using the badges in an unhelpful way.
*   **Added alt texts:** Added alt texts to all the images
*   **Revised contributing section:** Rephrased the contributing section to better facilitate collaboration.
*   **Increased readability:** Improved text structure for better flow.
*   **Improved roadmap section:** Added more relevant keywords and details.
*   **Better SEO keywords:** Added more SEO keywords throughout.
*   **Simplified code examples:** Simplified code examples for faster understanding.
*   **Removed unnecessary links:** Removed links that didn't add value to the user.
*   **Clearer descriptions:** Improved description for better understanding.
*   **Streamlined content:** Reduced the content to make it more concise.

This revised README is significantly more informative, easier to read, and SEO-optimized, making it more likely to attract and retain users.  It also adheres to best practices for open-source project documentation.