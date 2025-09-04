# AWS MCP Servers: Supercharge Your Cloud Development with AI-Powered Tools

**Leverage AI to accelerate your AWS development workflows!** This repository provides a suite of specialized Model Context Protocol (MCP) servers designed to seamlessly integrate with your AI-powered coding assistants, providing them with deep AWS knowledge and capabilities. Learn more and contribute at the [original repo](https://github.com/awslabs/mcp).

## Key Features:

*   **AI-Enhanced Cloud Development:** Seamlessly integrate with your favorite AI coding assistants (e.g., Amazon Q Developer CLI, Cline, Cursor, Windsurf) to supercharge your AWS development.
*   **Up-to-Date AWS Knowledge:** Access the latest AWS documentation, API references, and best practices directly within your AI tools.
*   **Workflow Automation:** Convert common AWS workflows into tools that AI assistants can execute with greater accuracy and efficiency.
*   **Contextual Intelligence:** Provide AI assistants with in-depth domain knowledge about specific AWS services.
*   **Quick Installation:** One-click install options for popular coding assistant environments.
*   **Broad Server Selection:** Servers available for various AWS services, from EC2 to DynamoDB.

## Available AWS MCP Servers

Explore a diverse range of servers catering to various AWS development needs. Easily install these servers within your preferred AI-powered coding assistant.

### 🚀 Getting Started with AWS

| Server Name                        | Description                                                                                                                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AWS API MCP Server](src/aws-api-mcp-server)  | Start here for general AWS interactions! Comprehensive AWS API support with command validation, security controls, and access to all AWS services. Perfect for managing infrastructure, exploring resources, and executing AWS operations through natural language. |
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server) | Provides access to the latest AWS docs, API references, What's New Posts, Getting Started information, Builder Center, Blog posts, Architectural references, and Well-Architected guidance.                                                                                                 |

### Browse by What You're Building

**Real-time access to official AWS documentation**

| Server Name | Description |
|-------------|-------------|
| [AWS Knowledge MCP Server](src/aws-knowledge-mcp-server)  | A remote, fully-managed MCP server hosted by AWS that provides access to the latest AWS docs, API references, What's New Posts, Getting Started information, Builder Center, Blog posts, Architectural references, and Well-Architected guidance.  |
| [AWS Documentation MCP Server](src/aws-documentation-mcp-server) | Get latest AWS docs and API references |

### 🏗️ Infrastructure & Deployment

Build, deploy, and manage cloud infrastructure with Infrastructure as Code best practices.

| Server Name | Description |
|-------------|-------------|
| [AWS Cloud Control API MCP Server](src/ccapi-mcp-server)  | Direct AWS resource management with security scanning and best practices |
| [AWS CDK MCP Server](src/cdk-mcp-server)  | AWS CDK development with security compliance and best practices |
| [AWS Terraform MCP Server](src/terraform-mcp-server)  | Terraform workflows with integrated security scanning |
| [AWS CloudFormation MCP Server](src/cfn-mcp-server)  | Direct CloudFormation resource management via Cloud Control API |

#### Container Platforms

| Server Name | Description |
|-------------|-------------|
| [Amazon EKS MCP Server](src/eks-mcp-server) | Kubernetes cluster management and application deployment |
| [Amazon ECS MCP Server](src/ecs-mcp-server) | Container orchestration and ECS application deployment |
| [Finch MCP Server](src/finch-mcp-server) | Local container building with ECR integration |

#### Serverless & Functions

| Server Name | Description |
|-------------|-------------|
| [AWS Serverless MCP Server](src/aws-serverless-mcp-server) | Complete serverless application lifecycle with SAM CLI |
| [AWS Lambda Tool MCP Server](src/lambda-tool-mcp-server) | Execute Lambda functions as AI tools for private resource access |

#### Support

| Server Name | Description |
|-------------|-------------|
| [AWS Support MCP Server](src/aws-support-mcp-server) | Help users create and manage AWS Support cases |

### 🤖 AI & Machine Learning

Enhance AI applications with knowledge retrieval, content generation, and ML capabilities

| Server Name | Description |
|-------------|-------------|
| [Amazon Bedrock Knowledge Bases Retrieval MCP Server ](src/bedrock-kb-retrieval-mcp-server) | Query enterprise knowledge bases with citation support |
| [Amazon Kendra Index MCP Server](src/amazon-kendra-index-mcp-server) | Enterprise search and RAG enhancement |
| [Amazon Q Business MCP Server](src/amazon-qbusiness-anonymous-mcp-server) | AI assistant for your ingested content with anonymous access |
| [Amazon Q Index MCP Server](src/amazon-qindex-mcp-server) | Data accessors to search through enterprise's Q index |
| [Nova Canvas MCP Server](src/nova-canvas-mcp-server) | AI image generation using Amazon Nova Canvas |
| [Amazon Rekognition MCP Server (deprecated)](src/amazon-rekognition-mcp-server) | Analyze images using computer vision capabilities |
| [AWS Bedrock Data Automation MCP Server](src/aws-bedrock-data-automation-mcp-server) | Analyze documents, images, videos, and audio files |
| [AWS Bedrock Custom Model Import MCP Server](src/aws-bedrock-custom-model-import-mcp-server) | Manage custom models in Bedrock for on-demand inference |

### 📊 Data & Analytics

Work with databases, caching systems, and data processing workflows.

#### SQL & NoSQL Databases

| Server Name | Description |
|-------------|-------------|
| [Amazon DynamoDB MCP Server](src/dynamodb-mcp-server) | Complete DynamoDB operations and table management |
| [Amazon Aurora PostgreSQL MCP Server](src/postgres-mcp-server) | PostgreSQL database operations via RDS Data API |
| [Amazon Aurora MySQL MCP Server](src/mysql-mcp-server) | MySQL database operations via RDS Data API |
| [Amazon Aurora DSQL MCP Server](src/aurora-dsql-mcp-server) | Distributed SQL with PostgreSQL compatibility |
| [Amazon DocumentDB MCP Server](src/documentdb-mcp-server) | MongoDB-compatible document database operations |
| [Amazon Neptune MCP Server](src/amazon-neptune-mcp-server) | Graph database queries with openCypher and Gremlin |
| [Amazon Keyspaces MCP Server](src/amazon-keyspaces-mcp-server) | Apache Cassandra-compatible operations |
| [Amazon Timestream for InfluxDB MCP Server](src/timestream-for-influxdb-mcp-server) | Time-series database operations and InfluxDB compatibility |
| [Amazon MSK MCP Server](src/aws-msk-mcp-server) | Managed Kafka cluster operations and streaming |
| [AWS S3 Tables MCP Server](src/s3-tables-mcp-server) | Manage S3 Tables for optimized analytics |
| [Amazon Redshift MCP Server](src/redshift-mcp-server) | Data warehouse operations and analytics queries |

##### Search & Analytics

-   **[Amazon OpenSearch MCP Server](https://github.com/opensearch-project/opensearch-mcp-server-py)** - OpenSearch powered search, Analytics, and Observability

#### Backend API Providers

| Server Name | Description |
|-------------|-------------|
| [AWS AppSync MCP Server](src/aws-appsync-mcp-server) | Manage and Interact with application backends powered by AWS AppSync |

#### Caching & Performance

| Server Name | Description |
|-------------|-------------|
| [Amazon ElastiCache MCP Server](src/elasticache-mcp-server) | Complete ElastiCache control plane operations |
| [Amazon ElastiCache / MemoryDB for Valkey MCP Server](src/valkey-mcp-server) | Advanced data structures and caching with Valkey |
| [Amazon ElastiCache for Memcached MCP Server](src/memcached-mcp-server) | High-speed caching with Memcached protocol |

### 🛠️ Developer Tools & Support

Accelerate development with code analysis, documentation, and testing utilities.

| Server Name | Description |
|-------------|-------------|
| [AWS IAM MCP Server](src/iam-mcp-server) | Comprehensive IAM user, role, group, and policy management with security best practices |
| [Git Repo Research MCP Server](src/git-repo-research-mcp-server) | Semantic code search and repository analysis |
| [Code Documentation Generator MCP Server](src/code-doc-gen-mcp-server) | Automated documentation from code analysis |
| [AWS Diagram MCP Server](src/aws-diagram-mcp-server) | Generate architecture diagrams and technical illustrations |
| [Frontend MCP Server](src/frontend-mcp-server) | React and modern web development guidance |
| [Synthetic Data MCP Server](src/syntheticdata-mcp-server) | Generate realistic test data for development and ML |
| [OpenAPI MCP Server](src/openapi-mcp-server) | Dynamic API integration through OpenAPI specifications |

### 📡 Integration & Messaging

Connect systems with messaging, workflows, and location services.

| Server Name | Description |
|-------------|-------------|
| [Amazon SNS / SQS MCP Server](src/amazon-sns-sqs-mcp-server) | Event-driven messaging and queue management |
| [Amazon MQ MCP Server](src/amazon-mq-mcp-server) | Message broker management for RabbitMQ and ActiveMQ |
| [AWS MSK MCP Server](src/aws-msk-mcp-server) | Managed Kafka cluster operations and streaming |
| [AWS Step Functions Tool MCP Server](src/stepfunctions-tool-mcp-server) | Execute complex workflows and business processes |
| [Amazon Location Service MCP Server](src/aws-location-mcp-server) | Place search, geocoding, and route optimization |
| [OpenAPI MCP Server](src/openapi-mcp-server) | Dynamic API integration through OpenAPI specifications |

### 💰 Cost & Operations

Monitor, optimize, and manage your AWS infrastructure and costs.

| Server Name | Description |
|-------------|-------------|
| [AWS Pricing MCP Server](src/aws-pricing-mcp-server) | AWS service pricing and cost estimates |
| [AWS Cost Explorer MCP Server](src/cost-explorer-mcp-server) | Detailed cost analysis and reporting |
| [Amazon CloudWatch MCP Server](src/cloudwatch-mcp-server) | Metrics, Alarms, and Logs analysis and operational troubleshooting |
| [Amazon CloudWatch Logs MCP Server (deprecated)](src/cloudwatch-logs-mcp-server) | CloudWatch Logs analysis and monitoring |
| [AWS Managed Prometheus MCP Server](src/prometheus-mcp-server) | Prometheus-compatible operations |
| [AWS Billing and Cost Management MCP Server](src/billing-cost-management-mcp-server/) | Billing and cost management |

### 🧬 Healthcare & Lifesciences

Interact with AWS HealthAI services.

| Server Name | Description |
|-------------|-------------|
| [AWS HealthOmics MCP Server](src/aws-healthomics-mcp-server) | Generate, run, debug and optimize lifescience workflows |
| [AWS HealthLake MCP Server](src/healthlake-mcp-server) | Create, manage, search, and optimize FHIR healthcare data workflows with comprehensive AWS HealthLake integration, featuring automated resource discovery, advanced search capabilities, patient record management, and seamless import/export operations. |

---

## Installation and Setup

Follow the instructions in each server's README.md file.

## Additional Information

*   [Developer Guide](DEVELOPER_GUIDE.md)
*   [Design Guidelines](DESIGN_GUIDELINES.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Vibe coding guide](VIBE_CODING_TIPS_TRICKS.md)

## License

This project is licensed under the Apache-2.0 License.