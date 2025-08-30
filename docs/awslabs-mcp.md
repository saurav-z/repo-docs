# AWS MCP Servers: Enhance AI-Powered Development on AWS

**Unlock the potential of your AI-powered tools with AWS MCP Servers, providing intelligent access to AWS services, documentation, and best practices.  Explore the original repo [here](https://github.com/awslabs/mcp).**

[![GitHub](https://img.shields.io/badge/github-awslabs/mcp-blue.svg?style=flat&logo=github)](https://github.com/awslabs/mcp)
[![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen)](LICENSE)
[![Codecov](https://img.shields.io/codecov/c/github/awslabs/mcp)](https://app.codecov.io/gh/awslabs/mcp)
[![OSSF-Scorecard Score](https://img.shields.io/ossf-scorecard/github.com/awslabs/mcp)](https://scorecard.dev/viewer/?uri=github.com/awslabs/mcp)

## Key Features:

*   **Enhanced AI Output**:  Receive more accurate, context-aware, and AWS-specific responses.
*   **Up-to-Date Documentation**: Access the latest AWS documentation, APIs, and SDKs directly within your AI workflow.
*   **Workflow Automation**: Transform common AWS tasks into executable AI tools, boosting efficiency.
*   **Specialized Knowledge**:  Gain in-depth, contextual knowledge about AWS services.

## Table of Contents

- [Key Features](#key-features)
- [Available Servers: Installation](#available-servers-installation)
- [Benefits of AWS MCP Servers](#benefits-of-aws-mcp-servers)
- [Use Cases](#use-cases)
- [Installation and Setup](#installation-and-setup)
- [Additional Resources](#additional-resources)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Available Servers: Installation

### 🚀 Get Started with AWS:

| Server Name | Description | Install |
|-------------|-------------|---------|
| [AWS API MCP Server](src/aws-api-mcp-server) | Comprehensive AWS API support for infrastructure management and exploration. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-api-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWFwaS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-api-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22type%22%3A%22stdio%22%7D) |
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | Access the latest AWS documentation, references, and best practices. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-knowledge-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWtub3dsZWRnZS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIiwiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D)<br/>[![Install VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

### Browse by What You're Building

[Add subsections for other server categories and tables like the above, including descriptions and one-click install for Cursor & VSCode.]

## Benefits of AWS MCP Servers

AWS MCP Servers empower foundation models (FMs) by:

*   Improving the quality of responses by integrating relevant AWS data and best practices.
*   Providing instant access to the most recent documentation for the latest AWS releases.
*   Enabling automation for common workflows into tools for FMs, such as CDK and Terraform.
*   Providing precise and helpful cloud development instructions by using specialized domain knowledge about AWS services.

## Use Cases

AWS MCP Servers streamline:

*   **Cloud-Native Development**: Use AI-powered tools to create and deploy applications.
*   **Infrastructure Management**: Control and manage your cloud infrastructure with natural language commands.
*   **Workflow Optimization**: Automate development, testing, and deployment processes.

[Add more specific examples, such as using the AWS Documentation MCP Server with a specific tool or task.]

## Installation and Setup

*   **Prerequisites**: Install `uv`, Python, and configure your AWS credentials.
*   **Configuration**: Edit your MCP client's configuration file (e.g., `~/.aws/amazonq/mcp.json` or `.vscode/mcp.json`) to add the server details.
*   **One-Click Installs**: Use the provided install buttons for Cursor and VS Code.

[Provide brief, clear instructions and examples for common MCP clients like Cursor, VS Code, and Amazon Q.]

## Additional Resources

*   [AWS Documentation]
*   [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
*   [Links to relevant AWS blog posts, videos, and tutorials].

## Security

For information about security, please refer to the [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) file.

## Contributing

Contributions are welcome!  See the [contributor guide](CONTRIBUTING.md) for details.

## License

This project is released under the Apache-2.0 License.