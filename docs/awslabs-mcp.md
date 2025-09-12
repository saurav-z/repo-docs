# AWS MCP Servers: Supercharge Your Cloud Development with AI

**Enhance your AI-powered development with specialized servers for the Model Context Protocol (MCP) and get the most out of AWS.**

[![GitHub](https://img.shields.io/badge/github-awslabs/mcp-blue.svg?style=flat&logo=github)](https://github.com/awslabs/mcp)
[![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen)](LICENSE)
[![Codecov](https://img.shields.io/codecov/c/github/awslabs/mcp)](https://app.codecov.io/gh/awslabs/mcp)
[![OSSF-Scorecard Score](https://img.shields.io/ossf-scorecard/github.com/awslabs/mcp)](https://scorecard.dev/viewer/?uri=github.com/awslabs/mcp)

## Key Features

*   **Improved Output Quality**: Enhance the accuracy and relevance of AI-generated responses with context-aware AWS knowledge.
*   **Access to Latest Documentation**: Stay up-to-date with the latest AWS services, APIs, and best practices.
*   **Workflow Automation**: Enable AI assistants to perform complex cloud tasks with greater efficiency.
*   **Specialized Domain Knowledge**: Leverage deep, contextual knowledge about AWS services.

## Table of Contents

*   [What is the Model Context Protocol (MCP)?](#what-is-the-model-context-protocol-mcp-and-how-does-it-work-with-aws-mcp-servers)
*   [Why AWS MCP Servers?](#why-aws-mcp-servers)
*   [Available MCP Servers](#available-mcp-servers-quick-installation)
    *   [🚀 Getting Started with AWS](#-getting-started-with-aws)
    *   [Browse by What You're Building](#browse-by-what-youre-building)
    *   [Browse by How You're Working](#browse-by-how-youre-working)
*   [MCP AWS Lambda Handler Module](#mcp-aws-lambda-handler-module)
*   [When to Use Local vs. Remote MCP Servers?](#when-to-use-local-vs-remote-mcp-servers)
*   [Use Cases for the Servers](#use-cases-for-the-servers)
*   [Installation and Setup](#installation-and-setup)
*   [Samples](#samples)
*   [Vibe Coding](#vibe-coding)
*   [Additional Resources](#additional-resources)
*   [Security](#security)
*   [Contributing](#contributing)
*   [Developer Guide](#developer-guide)
*   [License](#license)
*   [Disclaimer](#disclaimer)

## What is the Model Context Protocol (MCP) and how does it work with AWS MCP Servers?

>The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. Whether you're building an AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to connect LLMs with the context they need.
>
> &mdash; [Model Context Protocol README](https://github.com/modelcontextprotocol#:~:text=The%20Model%20Context,context%20they%20need.)

AWS MCP Servers leverage the Model Context Protocol (MCP) to provide your AI applications with intelligent access to AWS resources. MCP servers act as a bridge, delivering AWS documentation, best practices, and contextual guidance directly into your development environment and AI tools. This enables features like code generation, infrastructure management, and cloud-native development directly through your AI assistants.

*The Model Context Protocol is an open source project run by Anthropic, PBC. and open to contributions from the entire community. For more information on MCP, you can find further documentation [here](https://modelcontextprotocol.io/introduction)*

## Why AWS MCP Servers?

AWS MCP Servers empower foundation models (FMs) in several ways:

*   **Improved Output Quality**: Get more precise code and recommendations that align with current AWS capabilities.
*   **Access to the Latest Documentation**: Your AI assistant always works with the latest AWS capabilities and service updates.
*   **Workflow Automation**: Automate common workflows directly through your AI assistant.
*   **Specialized Domain Knowledge**: Get more helpful responses tailored to cloud development tasks.

## Available MCP Servers: Quick Installation

### 🚀 Getting Started with AWS

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [AWS API MCP Server](src/aws-api-mcp-server) | Comprehensive AWS API support for managing infrastructure. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-api-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWFwaS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-api-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22type%22%3A%22stdio%22%7D) |
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | Access the latest AWS docs and references. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-knowledge-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWtub3dsZWRnZS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIiwiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

### Browse by What You're Building

#### 📚 Real-time access to official AWS documentation

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | Access the latest AWS docs and references. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-knowledge-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWtub3dsZWRnZS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIiwiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Documentation MCP Server](src/aws-documentation-mcp-server) | Get the latest AWS documentation and API references | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-documentation-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWRvY3VtZW50YXRpb24tbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiIsIkFXU19ET0NVTUVOVEFUSU9OX1BBUlRJVElPTiI6ImF3cyJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D) <br/>[![Install on VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Documentation%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-documentation-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%2C%22AWS_DOCUMENTATION_PARTITION%22%3A%22aws%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### 🏗️ Infrastructure & Deployment

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [AWS Cloud Control API MCP Server](src/ccapi-mcp-server) | Direct AWS resource management with security scanning. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.ccapi-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuY2NhcGktbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiQVdTX1BST0ZJTEUiOiJ5b3VyLWF3cy1wcm9maWxlIiwiQVdTX1JFR0lPTiI6InVzLWVhc3QtMSIsIkZBU1RNQ1BfTE9HX0xFVkVMIjoiRVJST1IifSwiZGlzYWJsZWQiOmZhbHNlLCJhdXRvQXBwcm92ZSI6W119) <br/>[![Install on VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Cloud%20Control%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.ccapi-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS CDK MCP Server](src/cdk-mcp-server) | AWS CDK development with security and compliance. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.cdk-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuY2RrLW1jcC1zZXJ2ZXJAbGF0ZXN0IiwiZW52Ijp7IkZBU1RNQ1BfTE9HX0xFVkVMIjoiRVJST1IifSwiZGlzYWJsZWQiOmZhbHNlLCJhdXRvQXBwcm92ZSI6W119) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=CDK%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.cdk-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Terraform MCP Server](src/terraform-mcp-server) | Terraform workflows with integrated security scanning. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.terraform-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMudGVycmFmb3JtLW1jcC1zZXJ2ZXJAbGF0ZXN0IiwiZW52Ijp7IkZBU1RNQ1BfTE9HX0xFVkVMIjoiRVJST1IifSwiZGlzYWJsZWQiOmZhbHNlLCJhdXRvQXBwcm92ZSI6W119) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Terraform%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.terraform-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS CloudFormation MCP Server](src/cfn-mcp-server) | Direct CloudFormation resource management via Cloud Control API. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.cfn-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuY2ZuLW1jcC1zZXJ2ZXJAbGF0ZXN0IiwiZW52Ijp7IkFXU19QUk9GSUxFIjoieW91ci1uYW1lZC1wcm9maWxlIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=CloudFormation%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.cfn-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-named-profile%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### Container Platforms

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [Amazon EKS MCP Server](src/eks-mcp-server) | Kubernetes cluster management and deployment. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.eks-mcp-server&config=eyJhdXRvQXBwcm92ZSI6W10sImRpc2FibGVkIjpmYWxzZSwiY29tbWFuZCI6InV2eCBhd3NsYWJzLmVrcy1tY3Atc2VydmVyQGxhdGVzdCAtLWFsbG93LXdyaXRlIC0tYWxsb3ctc2Vuc2l0aXZlLWRhdGEtYWNjZXNzIiwiZW52Ijp7IkZBU1RNQ1BfTE9HX0xFVkVMIjoiRVJST1IifSwidHJhbnNwb3J0VHlwZSI6InN0ZGlvIn0%3D) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=EKS%20MCP%20Server&config=%7B%22autoApprove%22%3A%5B%5D%2C%22disabled%22%3Afalse%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.eks-mcp-server%40latest%22%2C%22--allow-write%22%2C%22--allow-sensitive-data-access%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22transportType%22%3A%22stdio%22%7D) |
| [Amazon ECS MCP Server](src/ecs-mcp-server) | Deploy containerized applications to ECS. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.ecs-mcp-server&config=eyJjb21tYW5kIjoidXZ4IC0tZnJvbSBhd3NsYWJzLWVjcy1tY3Atc2VydmVyIGVjcy1tY3Atc2VydmVyIiwiZW52Ijp7IkFXU19QUk9GSUxFIjoieW91ci1hd3MtcHJvZmlsZSIsIkFXU19SRUdJT04iOiJ5b3VyLWF3cy1yZWdpb24iLCJGQVNUTUNQX0xPR19MRVZFTCI6IkVSUk9SIiwiRkFTVE1DUF9MT0dfRklMRSI6Ii9wYXRoL3RvL2Vjcy1tY3Atc2VydmVyLmxvZyIsIkFMTE9XX1dSSVRFIjoiZmFsc2UiLCJBTExPV19TRU5TSVRJVkVfREFUQSI6ImZhbHNlIn19) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=ECS%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22--from%22%2C%22awslabs-ecs-mcp-server%22%2C%22ecs-mcp-server%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22your-aws-region%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%2C%22FASTMCP_LOG_FILE%22%3A%22%2Fpath%2Fto%2Fecs-mcp-server.log%22%2C%22ALLOW_WRITE%22%3A%22false%22%2C%22ALLOW_SENSITIVE_DATA%22%3A%22false%22%7D%7D) |
| [Finch MCP Server](src/finch-mcp-server) | Local container building with ECR integration. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.finch-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuZmluY2gtbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiQVdTX1BST0ZJTEUiOiJkZWZhdWx0IiwiQVdTX1JFR0lPTiI6InVzLXdlc3QtMiIsIkZBU1RNQ1BfTE9HX0xFVkVMIjoiSU5GTyJ9LCJ0cmFuc3BvcnRUeXBlIjoic3RkaW8iLCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Finch%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.finch-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22default%22%2C%22AWS_REGION%22%3A%22us-west-2%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22INFO%22%7D%2C%22transportType%22%3A%22stdio%22%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### Serverless & Functions

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [AWS Serverless MCP Server](src/aws-serverless-mcp-server) | Full serverless app lifecycle with SAM CLI. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-serverless-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLXNlcnZlcmxlc3MtbWNwLXNlcnZlckBsYXRlc3QgLS1hbGxvdy13cml0ZSAtLWFsbG93LXNlbnNpdGl2ZS1kYXRhLWFjY2VzcyIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Serverless%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-serverless-mcp-server%40latest%22%2C%22--allow-write%22%2C%22--allow-sensitive-data-access%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Lambda Tool MCP Server](src/lambda-tool-mcp-server) | Execute Lambda functions as AI tools for private resource access. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.lambda-tool-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMubGFtYmRhLXRvb2wtbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiQVdTX1BST0ZJTEUiOiJ5b3VyLWF3cy1wcm9maWxlIiwiQVdTX1JFR0lPTiI6InVzLWVhc3QtMSIsIkZVTkNUSU9OX1BSRUZJWCI6InlvdXItZnVuY3Rpb24tcHJlZml4IiwiRlVOQ1RJT05fTElTVCI6InlvdXItZmlyc3QtZnVuY3Rpb24sIHlvdXItc2Vjb25kLWZ1bmN0aW9uIiwiRlVOQ1RJT05fVEFHX0tFWSI6InlvdXItdGFnLWtleSIsIkZVTkNUSU9OX1RBR19WQUxVRSI6InlvdXItdGFnLXZhbHVlIiwiRlVOQ1RJT05fSU5QVVRfU0NIRU1BX0FSTl9UQUdfS0VZIjoieW91ci1mdW5jdGlvbi10YWctZm9yLWlucHV0LXNjaGVtYSJ9fQ%3D%3D) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Lambda%20Tool%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.lambda-tool-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FUNCTION_PREFIX%22%3A%22your-function-prefix%22%2C%22FUNCTION_LIST%22%3A%22your-first-function%2C%20your-second-function%22%2C%22FUNCTION_TAG_KEY%22%3A%22your-tag-key%22%2C%22FUNCTION_TAG_VALUE%22%3A%22your-tag-value%22%2C%22FUNCTION_INPUT_SCHEMA_ARN_TAG_KEY%22%3A%22your-function-tag-for-input-schema%22%7D%7D) |

#### Support

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [AWS Support MCP Server](src/aws-support-mcp-server) | Help users create and manage AWS Support cases. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs_support_mcp_server&config=eyJjb21tYW5kIjoidXZ4IC1tIGF3c2xhYnMuYXdzLXN1cHBvcnQtbWNwLXNlcnZlckBsYXRlc3QgLS1kZWJ1ZyAtLWxvZy1maWxlIC4vbG9ncy9tY3Bfc3VwcG9ydF9zZXJ2ZXIubG9nIiwiZW52Ijp7IkFXU19QUk9GSUxFIjoieW91ci1hd3MtcHJvZmlsZSJ9fQ%3D%3D) <br/>[![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Support%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22-m%22%2C%22awslabs.aws-support-mcp-server%40latest%22%2C%22--debug%22%2C%22--log-file%22%2C%22.%2Flogs%2Fmcp_support_server.log%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%7D%7D) |

### 🤖 AI & Machine Learning

| Server Name | Description | Install |
| :---------- | :---------- | :------ |
| [Amazon Bedrock Knowledge Bases Retrieval MCP Server](src