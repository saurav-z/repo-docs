# AWS MCP Servers: Enhance Your Cloud Development with AI

**Supercharge your AI-powered development experience on AWS with AWS MCP Servers!**

[![GitHub](https://img.shields.io/badge/github-awslabs/mcp-blue.svg?style=flat&logo=github)](https://github.com/awslabs/mcp)
[![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen)](LICENSE)
[![Codecov](https://img.shields.io/codecov/c/github/awslabs/mcp)](https://app.codecov.io/gh/awslabs/mcp)
[![OSSF-Scorecard Score](https://img.shields.io/ossf-scorecard/github.com/awslabs/mcp)](https://scorecard.dev/viewer/?uri=github.com/awslabs/mcp)

AWS MCP Servers provide specialized Model Context Protocol (MCP) servers to integrate AI-powered tools with AWS services, streamlining cloud-native development, infrastructure management, and workflows.  Explore the [original repo](https://github.com/awslabs/mcp) for comprehensive details and source code.

## Key Features

*   **AI-Enhanced Development:** Integrate AI coding assistants (e.g., Amazon Q Developer, Cline, Cursor) with AWS services.
*   **Comprehensive AWS Support:** Access to AWS documentation, APIs, and best practices directly within your AI workflows.
*   **Streamlined Workflows:** Automate complex tasks using tools optimized for AWS services like CDK and Terraform.
*   **Up-to-Date Information:**  Get the latest AWS documentation and API references, ensuring your AI assistant is always current.
*   **One-Click Installs:** Easily integrate with popular AI tools (Cursor, VS Code, and more).

## Important Notice: Server Sent Events (SSE) Support Removal

**Important Notice:** On May 26th, 2025, Server Sent Events (SSE) support was removed from all MCP servers in their latest major versions. This change aligns with the Model Context Protocol specification's [backwards compatibility guidelines](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#backwards-compatibility).

We are actively working towards supporting [Streamable HTTP](https://modelcontextprotocol.io/specification/draft/basic/transports#streamable-http), which will provide improved transport capabilities for future versions.

For applications still requiring SSE support, please use the previous major version of the respective MCP server until you can migrate to alternative transport methods.

## Why AWS MCP Servers?

AWS MCP Servers dramatically improve the capabilities of foundation models:

*   **Improved Output Quality:**  Provide relevant information in context for accurate technical details, precise code, and recommendations.
*   **Access to the Latest AWS Documentation:**  Bridge the knowledge gap by integrating with the most up-to-date AWS documentation.
*   **Workflow Automation:** Turn common workflows into tools that AI assistants can use directly.
*   **Specialized Domain Knowledge:**  Offer deep insights for more effective cloud development tasks.

## Available MCP Servers: Quick Installation

Get started with one-click installation buttons for popular MCP clients like Cursor and VS Code:

### 🚀 Getting Started with AWS

| Server Name | Description | Install |
|---|---|---|
| [AWS API MCP Server](src/aws-api-mcp-server) |  General AWS interactions! Comprehensive AWS API support with command validation, security controls, and access to all AWS services. |  [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-api-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWFwaS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-api-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22type%22%3A%22stdio%22%7D) |
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | Access the latest AWS docs, references, and guidance through a remote AWS-managed MCP server. |  [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-knowledge-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWtub3dsZWRnZS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIiwiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

### Browse by What You're Building

*   [📚 Real-time access to official AWS documentation](#-real-time-access-to-official-aws-documentation)
*   [🏗️ Infrastructure & Deployment](#️-infrastructure--deployment)
*   [🤖 AI & Machine Learning](#-ai--machine-learning)
*   [📊 Data & Analytics](#-data--analytics)
*   [🛠️ Developer Tools & Support](#️-developer-tools--support)
*   [📡 Integration & Messaging](#-integration--messaging)
*   [💰 Cost & Operations](#-cost--operations)
*   [🧬 Healthcare & Lifesciences](#-healthcare--lifesciences)

*(Detailed lists of servers within each category are provided in the original README.)*

### Browse by How You're Working

*   [👨‍💻 Vibe Coding & Development](#-vibe-coding--development)
*   [💬 Conversational Assistants](#-conversational-assistants)
*   [🤖 Autonomous Background Agents](#-autonomous-background-agents)

*(Detailed lists of servers within each category are provided in the original README.)*

## MCP AWS Lambda Handler Module

A Python library for creating serverless HTTP handlers for the Model Context Protocol (MCP) using AWS Lambda.

**Key Features:**

*   Simplifies creating serverless MCP HTTP handlers using AWS Lambda.
*   Offers session management, including built-in DynamoDB support.
*   Provides customizable authentication and authorization.

See the `src/mcp-lambda-handler/README.md` for full details.

## When to use Local vs Remote MCP Servers?

*   **Local:** For development, testing, data privacy, low latency, and resource control.
*   **Remote:** For team collaboration, resource-intensive tasks, high availability, automatic updates, and scalability.

## Use Cases

Enable AI assistants to research, generate code, automate infrastructure-as-code, perform cost analysis, and more.

## Installation and Setup

1.  Install `uv` from the link provided in "Getting Started with Cursor"
2.  Install Python using `uv python install 3.10`
3.  Configure AWS credentials.
4.  Add the server to your MCP client configuration (see the example for Amazon Q CLI MCP).

*   **Running in Containers:** Docker images are published to the public AWS ECR registry. Instructions provided.
*   **Getting Started Guides:**  Guides for Amazon Q Developer CLI, Kiro, Cline, Cursor, Windsurf, and VS Code are included.

## Samples

Explore ready-to-use examples in the [`samples/`](samples/) directory.

## Vibe Coding

Enhance your vibe coding experience with the help of the guide found [here](./VIBE_CODING_TIPS_TRICKS.md).

## Additional Resources

-   [Blog Posts, Videos, and more](https://aws.amazon.com/blogs/machine-learning/)

## Security

See [CONTRIBUTING.md](CONTRIBUTING.md#security-issue-notifications) for security guidelines.

## Contributing

Contributions are welcome! See the [contributor guide](CONTRIBUTING.md).

## Developer guide

Learn how to add new MCP Servers in the [development guide](DEVELOPER_GUIDE.md).

## License

This project is licensed under the Apache-2.0 License.

## Disclaimer

Review your security and quality control practices, along with all applicable laws, rules, and regulations before using MCP Servers.