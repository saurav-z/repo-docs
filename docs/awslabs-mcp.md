# Enhance Your AWS Development with AWS MCP Servers

**Unlock the power of AI-assisted cloud computing with AWS MCP Servers, seamlessly integrating with your development environment for enhanced productivity and efficiency.**

[![GitHub](https://img.shields.io/badge/github-awslabs/mcp-blue.svg?style=flat&logo=github)](https://github.com/awslabs/mcp)
[![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen)](LICENSE)
[![Codecov](https://img.shields.io/codecov/c/github/awslabs/mcp)](https://app.codecov.io/gh/awslabs/mcp)
[![OSSF-Scorecard Score](https://img.shields.io/ossf-scorecard/github.com/awslabs/mcp)](https://scorecard.dev/viewer/?uri=github.com/awslabs/mcp)

## Key Features:

*   **AI-Powered Development:** Integrate with AI-powered coding assistants to enhance cloud-native development.
*   **Real-time AWS Documentation:** Access up-to-date AWS documentation and API references.
*   **Infrastructure as Code:** Simplify infrastructure management with tools for CDK, Terraform, and CloudFormation.
*   **Container & Serverless Support:** Streamline container orchestration and serverless application deployment.
*   **Extensive AWS Service Coverage:** Support for a wide range of AWS services, including databases, AI/ML, and more.
*   **Vibe Coding Integration:** Leverage AI coding assistants for a more dynamic and interactive development experience.

## Table of Contents

-   [What are AWS MCP Servers?](#what-are-aws-mcp-servers)
-   [Why Use AWS MCP Servers?](#why-use-aws-mcp-servers)
-   [Available MCP Servers](#available-mcp-servers)
    -   [Getting Started](#getting-started)
    -   [Browse by Use Case](#browse-by-use-case)
        -   [Documentation and API References](#documentation-and-api-references)
        -   [Infrastructure & Deployment](#infrastructure--deployment)
        -   [Container Platforms](#container-platforms)
        -   [Serverless & Functions](#serverless--functions)
        -   [AI & Machine Learning](#ai--machine-learning)
        -   [Data & Analytics](#data--analytics)
        -   [Developer Tools & Support](#developer-tools--support)
        -   [Integration & Messaging](#integration--messaging)
        -   [Cost & Operations](#cost--operations)
        -   [Healthcare & Lifesciences](#healthcare--lifesciences)
    -   [Browse by Workflow](#browse-by-workflow)
        -   [Vibe Coding & Development](#vibe-coding--development)
        -   [Conversational Assistants](#conversational-assistants)
        -   [Autonomous Background Agents](#autonomous-background-agents)
-   [Installation and Setup](#installation-and-setup)
    -   [Amazon Q Developer CLI](#getting-started-with-amazon-q-developer-cli)
    -   [Kiro](#getting-started-with-kiro)
    -   [Cline and Amazon Bedrock](#getting-started-with-cline-and-amazon-bedrock)
    -   [Cursor](#getting-started-with-cursor)
    -   [Windsurf](#getting-started-with-windsurf)
    -   [VS Code](#getting-started-with-vs-code)
-   [Samples](#samples)
-   [Additional Resources](#additional-resources)
-   [Security](#security)
-   [Contributing](#contributing)
-   [Developer Guide](#developer-guide)
-   [License](#license)

## What are AWS MCP Servers?

AWS MCP Servers are specialized servers built upon the Model Context Protocol (MCP) designed to enhance your development experience with AWS. These servers provide AI applications, such as coding assistants and chat interfaces, with access to AWS documentation, contextual guidance, and best practices. By leveraging the MCP client-server architecture, AWS capabilities become an intelligent extension of your development environment. For more information on MCP, visit the [original repo](https://github.com/awslabs/mcp).

## Why Use AWS MCP Servers?

AWS MCP Servers significantly improve the capabilities of foundation models (FMs) in the following ways:

*   **Improved Output Quality:** Offer enhanced model responses for specialized domains like AWS services.
*   **Access to Latest Documentation:** Ensure that your AI assistant always works with the latest AWS capabilities.
*   **Workflow Automation:** Convert common workflows into tools that foundation models can use directly.
*   **Specialized Domain Knowledge:** Provide deep, contextual knowledge about AWS services.

## Available MCP Servers

Explore a range of AWS MCP Servers tailored to different use cases:

### Getting Started

For general AWS interactions and comprehensive API support, we recommend starting with:

| Server Name | Description | Install |
|-------------|-------------|---------|
| [AWS API MCP Server](src/aws-api-mcp-server) | Start here for general AWS interactions! Comprehensive AWS API support with command validation, security controls, and access to all AWS services. Perfect for managing infrastructure, exploring resources, and executing AWS operations through natural language. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-api-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWFwaS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-api-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22type%22%3A%22stdio%22%7D) |
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | A remote, fully-managed MCP server hosted by AWS that provides access to the latest AWS docs, API references, What's New Posts, Getting Started information, Builder Center, Blog posts, Architectural references, and Well-Architected guidance. | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-knowledge-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWtub3dsZWRnZS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIiwiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D)<br/>[![Install VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

### Browse by Use Case

#### Documentation and API References

| Server Name | Description | Install |
|-------------|-------------|---------|
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | Provides access to the latest AWS documentation, API references, and more. |  [![Install](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Documentation MCP Server](src/aws-documentation-mcp-server) | Get latest AWS docs and API references |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Documentation%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-documentation-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%2C%22AWS_DOCUMENTATION_PARTITION%22%3A%22aws%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### Infrastructure & Deployment

| Server Name | Description | Install |
|-------------|-------------|---------|
| [AWS Cloud Control API MCP Server](src/ccapi-mcp-server) | Direct AWS resource management. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Cloud%20Control%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.ccapi-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS CDK MCP Server](src/cdk-mcp-server) | For AWS CDK development. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=CDK%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.cdk-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Terraform MCP Server](src/terraform-mcp-server) | For Terraform workflows. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Terraform%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.terraform-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS CloudFormation MCP Server](src/cfn-mcp-server) | CloudFormation resource management. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=CloudFormation%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.cfn-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-named-profile%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### Container Platforms

| Server Name | Description | Install |
|-------------|-------------|---------|
| [Amazon EKS MCP Server](src/eks-mcp-server) | For Kubernetes cluster management and application deployment. | [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=EKS%20MCP%20Server&config=%7B%22autoApprove%22%3A%5B%5D%2C%22disabled%22%3Afalse%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.eks-mcp-server%40latest%22%2C%22--allow-write%22%2C%22--allow-sensitive-data-access%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22transportType%22%3A%22stdio%22%7D) |
| [Amazon ECS MCP Server](src/ecs-mcp-server) | Container orchestration and deployment to ECS. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=ECS%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22--from%22%2C%22awslabs-ecs-mcp-server%22%2C%22ecs-mcp-server%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22your-aws-region%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%2C%22FASTMCP_LOG_FILE%22%3A%22%2Fpath%2Fto%2Fecs-mcp-server.log%22%2C%22ALLOW_WRITE%22%3A%22false%22%2C%22ALLOW_SENSITIVE_DATA%22%3A%22false%22%7D%7D) |
| [Finch MCP Server](src/finch-mcp-server) | Local container building with ECR integration |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Finch%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.finch-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22default%22%2C%22AWS_REGION%22%3A%22us-west-2%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22INFO%22%7D%2C%22transportType%22%3A%22stdio%22%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### Serverless & Functions

| Server Name | Description | Install |
|-------------|-------------|---------|
| [AWS Serverless MCP Server](src/aws-serverless-mcp-server) | For complete serverless application lifecycle with SAM CLI. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Serverless%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-serverless-mcp-server%40latest%22%2C%22--allow-write%22%2C%22--allow-sensitive-data-access%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Lambda Tool MCP Server](src/lambda-tool-mcp-server) | Execute Lambda functions as AI tools. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Lambda%20Tool%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.lambda-tool-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FUNCTION_PREFIX%22%3A%22your-function-prefix%22%2C%22FUNCTION_LIST%22%3A%22your-first-function%2C%20your-second-function%22%2C%22FUNCTION_TAG_KEY%22%3A%22your-tag-key%22%2C%22FUNCTION_TAG_VALUE%22%3A%22your-tag-value%22%2C%22FUNCTION_INPUT_SCHEMA_ARN_TAG_KEY%22%3A%22your-function-tag-for-input-schema%22%7D%7D) |

#### AI & Machine Learning

| Server Name | Description | Install |
|-------------|-------------|---------|
| [Amazon Bedrock Knowledge Bases Retrieval MCP Server ](src/bedrock-kb-retrieval-mcp-server) | Query enterprise knowledge bases with citation support |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Bedrock%20KB%20Retrieval%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.bedrock-kb-retrieval-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-profile-name%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%2C%22KB_INCLUSION_TAG_KEY%22%3A%22optional-tag-key-to-filter-kbs%22%2C%22BEDROCK_KB_RERANKING_ENABLED%22%3A%22false%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [Amazon Kendra Index MCP Server](src/amazon-kendra-index-mcp-server) | Enterprise search and RAG enhancement |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Amazon%20Kendra%20Index%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.amazon-kendra-index-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%2C%22KEND_INDEX_ID%22%3A%22your-kendra-index-id%22%2C%22KEND_ROLE_ARN%22%3A%22your-kendra-role-arn%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [Amazon Q Business MCP Server](src/amazon-qbusiness-anonymous-mcp-server) | AI assistant for your ingested content with anonymous access |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Amazon%20Q%20Business%20Anonymous%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.amazon-qbusiness-anonymous-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22QBUSINESS_APP_ID%22%3A%22your-qbusiness-app-id%22%2C%22QBUSINESS_USER_ID%22%3A%22your-user-id%22%2C%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [Amazon Q Index MCP Server](src/amazon-qindex-mcp-server) | Data accessors to search through enterprise's Q index |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Amazon%20Q%20Index%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.amazon-qindex-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%2C%22QINDEX_ID%22%3A%22your-qindex-id%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [Nova Canvas MCP Server](src/nova-canvas-mcp-server) | AI image generation using Amazon Nova Canvas |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Nova%20Canvas%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.nova-canvas-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [Amazon Rekognition MCP Server (deprecated)](src/amazon-rekognition-mcp-server) | Analyze images using computer vision capabilities |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Amazon%20Rekognition%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.amazon-rekognition-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Bedrock Data Automation MCP Server](src/aws-bedrock-data-automation-mcp-server) | Analyze documents, images, videos, and audio files |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=Bedrock%20Data%20Automation%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-bedrock-data-automation-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22AWS_BUCKET_NAME%22%3A%22your-s3-bucket-name%22%2C%22BASE_DIR%22%3A%22%2Fpath%2Fto%2Fbase%2Fdirectory%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [AWS Bedrock Custom Model Import MCP Server](src/aws-bedrock-custom-model-import-mcp-server) | Manage custom models in Bedrock for on-demand inference |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Bedrock%20Custom%20Model%20Import%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-bedrock-custom-model-import-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22BEDROCK_MODEL_IMPORT_S3_BUCKET%22%3A%22your-s3-bucket-name%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

#### Data & Analytics

##### SQL & NoSQL Databases

| Server Name | Description | Install |
|-------------|-------------|---------|
| [Amazon DynamoDB MCP Server](src/dynamodb-mcp-server) | Complete DynamoDB operations. |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=DynamoDB%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.dynamodb-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22DDB-MCP-READONLY%22%3A%22true%22%2C%22AWS_PROFILE%22%3A%22default%22%2C%22AWS_REGION%22%3A%22us-west-2%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |
| [Amazon Aurora PostgreSQL MCP Server](src/postgres-mcp-server) | PostgreSQL database operations via RDS Data API |  [![Install on VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=PostgreSQL%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.postgres-mcp-server%40latest%22%2C%22--connection-string%22%2C%22postgresql%3A%2F%2F%5Busername%5D%3A%5Bpassword%5D%40%5Bhost%5D%3A%5Bport%5D%2F%5Bdatabase%5D%22%5D%2C%22env%22%3A%7B%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22