# Supercharge Your AWS Development with AWS MCP Servers

**Unlock the power of AI-assisted development and streamline your cloud workflows with AWS MCP Servers, [the official repository for AWS MCP servers](https://github.com/awslabs/mcp).**

[![GitHub](https://img.shields.io/badge/github-awslabs/mcp-blue.svg?style=flat&logo=github)](https://github.com/awslabs/mcp)
[![License](https://img.shields.io/badge/license-Apache--2.0-brightgreen)](LICENSE)
[![Codecov](https://img.shields.io/codecov/c/github/awslabs/mcp)](https://app.codecov.io/gh/awslabs/mcp)
[![OSSF-Scorecard Score](https://img.shields.io/ossf-scorecard/github.com/awslabs/mcp)](https://scorecard.dev/viewer/?uri=github.com/awslabs/mcp)

## Key Features

*   **AI-Powered Development:** Integrate directly with your favorite IDEs and AI coding assistants to get accurate code generation, contextual guidance, and automated workflows.
*   **Real-time AWS Knowledge:** Access the latest AWS documentation, best practices, and API references directly within your development environment.
*   **Streamlined Workflows:** Automate common cloud tasks and manage infrastructure with natural language commands and integrated tools.
*   **Enhanced Cloud Native Development:** Take advantage of the standardized Model Context Protocol (MCP) to enhance cloud-native development, infrastructure management, and development workflows.
*   **Wide Range of Servers:** Choose from a growing library of specialized MCP servers tailored for various AWS services and use cases.

## Why AWS MCP Servers?

AWS MCP Servers provide critical enhancements to foundation models (FMs):

*   **Improved Output Quality**: Receive more accurate, domain-specific responses, reducing hallucinations.
*   **Up-to-Date Documentation**: Access real-time AWS documentation for accurate, up-to-date information.
*   **Automated Workflows**: Streamline complex tasks by leveraging server tools for greater efficiency and accuracy.
*   **Specialized Knowledge**: Benefit from contextual AWS service knowledge to improve cloud-native development tasks.

## Available MCP Servers

### 🚀 Getting Started with AWS

| Server Name               | Description                                                                                                                                                                                                                                                                                                                              | Install                                                                                                                                                                                                                                                                                                                                                                                           |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AWS API MCP Server**     | Start here for general AWS interactions! Comprehensive AWS API support with command validation, security controls, and access to all AWS services. Perfect for managing infrastructure, exploring resources, and executing AWS operations through natural language.                                                                         | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-api-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWFwaS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIn0sImRpc2FibGVkIjpmYWxzZSwiYXV0b0FwcHJvdmUiOltdfQ%3D%3D)<br/>[![Install VS Code](https://img.shields.io/badge/Install-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20API%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-api-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_REGION%22%3A%22us-east-1%22%7D%2C%22type%22%3A%22stdio%22%7D) |
| **AWS Knowledge MCP Server** | A remote, fully-managed MCP server hosted by AWS that provides access to the latest AWS docs, API references, What's New Posts, Getting Started information, Builder Center, Blog posts, Architectural references, and Well-Architected guidance.                                                                                 | [![Install](https://img.shields.io/badge/Install-Cursor-blue?style=flat-square&logo=cursor)](https://cursor.com/en/install-mcp?name=awslabs.aws-knowledge-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYXdzLWtub3dsZWRnZS1tY3Atc2VydmVyQGxhdGVzdCIsImVudiI6eyJBV1NfUFJPRklMRSI6InlvdXItYXdzLXByb2ZpbGUiLCJBV1NfUkVHSU9OIjoidXMtZWFzdC0xIiwiRkFTVE1DUF9MT0dfTEVWRUwiOiJFUlJPUiJ9LCJkaXNhYmxlZCI6ZmFsc2UsImF1dG9BcHByb3ZlIjpbXX0%3D)<br/>[![Install VS Code](https://insiders.vscode.dev/redirect/mcp/install?name=AWS%20Knowledge%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.aws-knowledge-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

### Browse by What You're Building

#### 📚 Real-time access to official AWS documentation

*   **AWS Knowledge MCP Server:** Get access to AWS docs, API references, What's New Posts, getting started guides, blogs, and more!
*   **AWS Documentation MCP Server:** Get latest AWS docs and API references

#### 🏗️ Infrastructure & Deployment

*   **AWS CDK MCP Server:** CDK development with security best practices and compliance
*   **AWS Terraform MCP Server:** Terraform with integrated security scanning and best practices
*   **AWS CloudFormation MCP Server:** Direct AWS resource management through Cloud Control API
*   **AWS Cloud Control API MCP Server:** Direct AWS resource management with security scanning and best practices

#### Container Platforms

*   **Amazon EKS MCP Server:** Kubernetes cluster management and app deployment
*   **Amazon ECS MCP Server:** Containerize and deploy applications to ECS
*   **Finch MCP Server:** Local container building with ECR push

#### Serverless & Functions

*   **AWS Serverless MCP Server:** Full serverless app lifecycle with SAM CLI
*   **AWS Lambda Tool MCP Server:** Execute Lambda functions as AI tools for private resource access

#### Support

*   **AWS Support MCP Server:** Help users create and manage AWS Support cases

#### 🤖 AI & Machine Learning

*   **Amazon Bedrock Knowledge Bases Retrieval MCP Server:** Query enterprise knowledge bases with citation support
*   **Amazon Kendra Index MCP Server:** Enterprise search and RAG enhancement
*   **Amazon Q Business MCP Server:** AI assistant for your ingested content with anonymous access
*   **Amazon Q Index MCP Server:** Data accessors to search through enterprise's Q index
*   **Nova Canvas MCP Server:** Generate images from text descriptions and color palettes
*   **Amazon Rekognition MCP Server (deprecated):** Analyze images using computer vision capabilities
*   **AWS Bedrock Data Automation MCP Server:** Analyze uploaded documents, images, and media
*   **AWS Bedrock Custom Model Import MCP Server:** Manage custom models in Bedrock for on-demand inference

#### 📊 Data & Analytics

##### SQL & NoSQL Databases

*   **Amazon DynamoDB MCP Server:** Complete DynamoDB operations and table management
*   **Amazon Aurora PostgreSQL MCP Server:** PostgreSQL database operations via RDS Data API
*   **Amazon Aurora MySQL MCP Server:** MySQL database operations via RDS Data API
*   **Amazon Aurora DSQL MCP Server:** Distributed SQL with PostgreSQL compatibility
*   **Amazon DocumentDB MCP Server:** MongoDB-compatible document database operations
*   **Amazon Neptune MCP Server:** Graph database queries with openCypher and Gremlin
*   **Amazon Keyspaces MCP Server:** Apache Cassandra-compatible operations
*   **Amazon Timestream for InfluxDB MCP Server:** Time-series database operations and InfluxDB compatibility
*   **Amazon MSK MCP Server:** Managed Kafka cluster operations and streaming
*   **AWS S3 Tables MCP Server:** Manage S3 Tables for optimized analytics
*   **Amazon Redshift MCP Server:** Data warehouse operations and analytics queries

##### Search & Analytics

*   **[Amazon OpenSearch MCP Server](https://github.com/opensearch-project/opensearch-mcp-server-py):** OpenSearch-powered search, Analytics, and Observability

##### Caching & Performance

*   **Amazon ElastiCache MCP Server:** Complete ElastiCache control plane operations
*   **Amazon ElastiCache / MemoryDB for Valkey MCP Server:** Advanced data structures and caching with Valkey
*   **Amazon ElastiCache for Memcached MCP Server:** High-speed caching with Memcached protocol

#### 🛠️ Developer Tools & Support

*   **AWS IAM MCP Server:** Comprehensive IAM user, role, group, and policy management with security best practices
*   **Git Repo Research MCP Server:** Semantic code search and repository analysis
*   **Code Documentation Generator MCP Server:** Automated documentation from code analysis
*   **AWS Diagram MCP Server:** Generate architecture diagrams and technical illustrations
*   **Frontend MCP Server:** React and modern web development guidance
*   **Synthetic Data MCP Server:** Generate realistic test data for development and ML
*   **OpenAPI MCP Server:** Dynamic API integration through OpenAPI specifications

#### 📡 Integration & Messaging

*   **Amazon SNS/SQS MCP Server:** Event-driven messaging and queue management
*   **Amazon MQ MCP Server:** Message broker management for RabbitMQ and ActiveMQ
*   **AWS MSK MCP Server:** Managed Kafka cluster operations and streaming
*   **AWS Step Functions Tool MCP Server:** Execute complex workflows and business processes
*   **Amazon Location Service MCP Server:** Place search, geocoding, and route optimization
*   **OpenAPI MCP Server:** Dynamic API integration through OpenAPI specifications

#### 💰 Cost & Operations

*   **AWS Pricing MCP Server:** AWS service pricing and cost estimates
*   **AWS Cost Explorer MCP Server:** Detailed cost analysis and reporting
*   **Amazon CloudWatch MCP Server:** Metrics, Alarms, and Logs analysis and operational troubleshooting
*   **Amazon CloudWatch Logs MCP Server (deprecated):** CloudWatch Logs analysis and monitoring
*   **AWS Managed Prometheus MCP Server:** Prometheus-compatible operations and monitoring
*   **AWS Billing and Cost Management MCP Server:** Billing and cost management

#### 🧬 Healthcare & Lifesciences

*   **AWS HealthOmics MCP Server:** Generate, run, debug and optimize lifescience workflows
*   **AWS HealthLake MCP Server:** Create, manage, search, and optimize FHIR healthcare data workflows

---
---

### Browse by How You're Working

#### 👨‍💻 Vibe Coding & Development

*AI coding assistants like Amazon Q Developer CLI, Cline, Cursor, and Claude Code helping you build faster*

##### Core Development Workflow

*   **AWS API MCP Server:** Comprehensive AWS API support with command validation, security controls, and access to all AWS services.
*   **Core MCP Server:** Intelligent planning and MCP server orchestration
*   **AWS Knowledge MCP Server:** Access to the latest AWS docs, API references, What's New Posts, Getting Started information, Builder Center, Blog posts, Architectural references, and Well-Architected guidance.
*   **AWS Documentation MCP Server:** Get latest AWS docs and API references
*   **Git Repo Research MCP Server:** Semantic search through codebases and repositories

##### Infrastructure as Code

*   **AWS CDK MCP Server:** CDK development with security best practices and compliance
*   **AWS Terraform MCP Server:** Terraform with integrated security scanning and best practices
*   **AWS CloudFormation MCP Server:** Direct AWS resource management through Cloud Control API
*   **AWS Cloud Control API MCP Server:** Direct AWS resource management with security scanning and best practices

##### Application Development

*   **Frontend MCP Server:** React and modern web development patterns with AWS integration
*   **AWS Diagram MCP Server:** Generate architecture diagrams as you design
*   **Code Documentation Generation MCP Server:** Auto-generate docs from your codebase
*   **OpenAPI MCP Server:** Dynamic API integration through OpenAPI specifications

##### Container & Serverless Development

*   **Amazon EKS MCP Server:** Kubernetes cluster management and app deployment
*   **Amazon ECS MCP Server:** Containerize and deploy applications to ECS
*   **Finch MCP Server:** Local container building with ECR push
*   **AWS Serverless MCP Server:** Full serverless app lifecycle with SAM CLI

##### Testing & Data

*   **Synthetic Data MCP Server:** Generate realistic test data for development and ML

##### Lifesciences Workflow Development

*   **AWS HealthOmics MCP Server:** Generate, run, debug and optimize lifescience workflows

##### Healthcare Data Management

*   **[AWS HealthLake MCP Server](src/healthlake-mcp-server):** Create, manage, search, and optimize FHIR healthcare data workflows

#### 💬 Conversational Assistants

*Customer-facing chatbots, business agents, and interactive Q&A systems*

##### Knowledge & Search

*   **Amazon Bedrock Knowledge Bases Retrieval MCP Server:** Query enterprise knowledge bases with citation support
*   **Amazon Kendra Index MCP Server:** Enterprise search and RAG enhancement
*   **Amazon Q Business MCP Server:** AI assistant for your ingested content with anonymous access
*   **Amazon Q Index MCP Server:** Data accessors to search through enterprise's Q index
*   **AWS Documentation MCP Server:** Get latest AWS docs and API references

##### Content Processing & Generation

*   **Amazon Nova Canvas MCP Server:** Generate images from text descriptions and color palettes
*   **Amazon Rekognition MCP Server (deprecated):** Analyze images using computer vision capabilities
*   **Amazon Bedrock Data Automation MCP Server:** Analyze uploaded documents, images, and media

##### Business Services

*   **Amazon Location Service MCP Server:** Location search, geocoding, and business hours
*   **AWS Pricing MCP Server:** AWS service pricing and cost estimates
*   **AWS Cost Explorer MCP Server:** Detailed cost analysis and spend reports

#### 🤖 Autonomous Background Agents

*Headless automation, ETL pipelines, and operational systems*

##### Data Operations & ETL

*   **AWS Data Processing MCP Server:** Comprehensive data processing tools and real-time pipeline visibility across AWS Glue and Amazon EMR-EC2
*   **Amazon DynamoDB MCP Server:** Complete DynamoDB operations and table management
*   **Amazon Aurora PostgreSQL MCP Server:** PostgreSQL database operations via RDS Data API
*   **Amazon Aurora MySQL MCP Server:** MySQL database operations via RDS Data API
*   **Amazon Aurora DSQL MCP Server:** Distributed SQL with PostgreSQL compatibility
*   **Amazon DocumentDB MCP Server:** MongoDB-compatible document database operations
*   **Amazon Neptune MCP Server:** Graph database queries with openCypher and Gremlin
*   **Amazon Keyspaces MCP Server:** Apache Cassandra-compatible operations
*   **Amazon Timestream for InfluxDB MCP Server:** Time-series database operations and InfluxDB compatibility
*   **Amazon MSK MCP Server:** Managed Kafka cluster operations and streaming

##### Caching & Performance

*   **Amazon ElastiCache / MemoryDB for Valkey MCP Server:** Advanced data structures and caching with Valkey
*   **Amazon ElastiCache for Memcached MCP Server:** High-speed caching with Memcached protocol

##### Workflow & Integration

*   **AWS Lambda Tool MCP Server:** Execute Lambda functions as AI tools for private resource access
*   **AWS Step Functions Tool MCP Server:** Execute complex workflows and business processes
*   **Amazon SNS/SQS MCP Server:** Event-driven messaging and queue management
*   **Amazon MQ MCP Server:** Message broker management for RabbitMQ and ActiveMQ
*   **AWS MSK MCP Server:** Managed Kafka cluster operations and streaming

##### Operations & Monitoring

*   **Amazon CloudWatch MCP Server:** Metrics, Alarms, and Logs analysis and operational troubleshooting
*   **Amazon CloudWatch Logs MCP Server (deprecated):** CloudWatch Logs analysis and monitoring
*   **Amazon CloudWatch Application Signals MCP Server:** Application monitoring and performance insights
*   **AWS Cost Explorer MCP Server:** Detailed cost analysis and reporting
*   **AWS Managed Prometheus MCP Server:** Prometheus-compatible operations and monitoring
*   **AWS Well-Architected Security Assessment Tool MCP Server:** Assess AWS environments against the Well-Architected Framework Security Pillar
*   **AWS CloudTrail MCP Server:** CloudTrail events querying and analysis

## MCP AWS Lambda Handler Module

A Python library is available for creating serverless HTTP handlers for the Model Context Protocol (MCP) using AWS Lambda. This module provides a flexible framework for building MCP HTTP endpoints with pluggable session management, including built-in DynamoDB support.

**Features:**

-   Easy serverless MCP HTTP handler creation using AWS Lambda
-   Pluggable session management system
-   Built-in DynamoDB session backend support
-   Customizable authentication and authorization
-   Example implementations and tests

See [`src/mcp-lambda-handler/README.md`](src/mcp-lambda-handler/README.md) for full usage, installation, and development instructions.

## When to use Local vs Remote MCP Servers?

Choose the right deployment strategy for your needs:

### Local MCP Servers

*   **Development & Testing:** Ideal for local development, testing, and debugging.
*   **Offline Work:** Continue working even with limited internet connectivity.
*   **Data Privacy:** Securely manage sensitive data and credentials locally.
*   **Low Latency:** Enjoy faster response times with minimal network overhead.
*   **Resource Control:** Maintain direct control over server resources.

### Remote MCP Servers

*   **Team Collaboration:** Share consistent server configurations across your team.
*   **Resource Intensive Tasks:** Offload heavy processing to dedicated cloud resources.
*   **Always Available:** Ensure continuous access to your MCP servers.
*   **Automatic Updates:** Get the latest features and security patches.
*   **Scalability:** Easily handle varying workloads without local limitations.

>   **Note:** Some MCP servers, like AWS Knowledge MCP, are fully managed services, removing the need for local setup.

## Installation and Setup

1.  Install `uv` from [Astral](https://docs.astral.sh/uv/getting-started/installation/)
2.  Install Python using `uv python install 3.10`
3.  Configure AWS credentials with access to required services.
4.  Integrate the server with your chosen MCP client via configuration files.

## Samples

Explore pre-built examples demonstrating the use of each server, located in the [samples](samples/) directory, to help you get started.

## Vibe Coding

Enhance your coding experience with AI assistants using the AWS MCP servers. For tips, tricks and guides, please refer to our [vibe coding guide](./VIBE_CODING_TIPS_TRICKS.md).

## Additional Resources

*   [Introducing AWS MCP Servers for code assistants](https://aws.amazon.com/blogs/machine-learning/introducing-aws-mcp-servers-for-code-assistants-part-1/)
*   [Vibe coding with AWS MCP Servers | AWS Show & Tell](https://www.youtube.com/watch?v=qXGQQRMrcz0)
*   [Supercharging AWS database development with AWS MCP servers](https://aws.amazon.com/blogs/database/supercharging-aws-database-development-with-aws-mcp-servers/)
*   [AWS costs estimation using Amazon Q CLI and AWS Pricing MCP Server](https://aws.amazon.com/blogs/machine-learning/aws-costs-estimation-using-amazon-q-cli-and-aws-cost-analysis-mcp/)
*   [Introducing AWS Serverless MCP Server: AI-powered development for modern applications](https://aws.amazon.com/blogs/compute/introducing-aws-serverless-mcp-server-ai-powered-development-for-modern-applications/)
*   [Announcing new Model Context Protocol (MCP) Servers for AWS Serverless and Containers](https://aws.amazon.com/about-aws/whats-new/2025/05/new-model-context-protocol-servers-aws-serverless-containers/)
*   [Accelerating application development with the Amazon EKS MCP server](https://aws.amazon.com/blogs/containers/accelerating-application-development-with-the-amazon-eks-model-context-protocol-server/)
*   [Amazon Neptune announces MCP (Model Context Protocol) Server](https://aws.amazon.com/about-aws/whats-new/2025/05/amazon-neptune-mcp-server/)
*   [Terraform MCP Server Vibe Coding](https://youtu.be/i2nBD65md0Y)
*   [How to Generate AWS Architecture Diagrams Using Amazon Q CLI and MCP](https://community.aws/content/2vPiiPiBSdRalaEax2rVDtshpf3/how-to-generate-aws-architecture-diagrams-using-amazon-q-cli-and-mcp)
*   [Harness the power of MCP servers with Amazon Bedrock Agents](https://aws.amazon.com/blogs/machine-learning/harness-the-power-of-mcp-servers-with-amazon-bedrock-agents/)
*   [Unlocking the power of Model Context Protocol (MCP) on AWS](https://aws.amazon.com/blogs/machine-learning/unlocking-the-power-of-model-context-protocol-mcp-on-aws/)
*   [AWS Price List Gets a Natural Language Upgrade: Introducing the AWS Pricing MCP Server](https://aws.amazon.com/blogs/aws-cloud-financial-management/aws-price-list-gets-a-natural-language-upgrade-introducing-the-aws-pricing-mcp-server/)
*   [AWS SheBuilds: AWS Team's Journey from Internal Tools to Open Source AI Infrastructure](https://www.youtube.com/watch?v=DZFgufNCvAo)

## Security

Refer to [CONTRIBUTING.md#security-issue-notifications](CONTRIBUTING.md#security-issue-notifications) for information.

## Contributing

Contribute and help make this project better! See our [contributor guide](CONTRIBUTING.md) for more information.

[![contributors](https://contrib.rocks/image?repo=awslabs/mcp&max=2000)](https://github.com/awslabs/mcp/graphs/contributors)

## Developer Guide

If you're interested in contributing a new MCP Server, check out our [development guide](DEVELOPER_GUIDE.md) and [design guidelines](DESIGN_GUIDELINES.md).

## License

This project is licensed under the Apache-2.0 License.

## Disclaimer

Before using an MCP Server, always conduct your own independent assessment to confirm the use aligns with your specific security, quality control practices, and applicable legal requirements.