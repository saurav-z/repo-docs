[![Metaflow Logo](https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png)](https://github.com/Netflix/metaflow)

# Metaflow: Build, Manage, and Scale Your AI/ML Systems

**Metaflow, originally developed at Netflix, is a human-centric framework designed to empower data scientists and engineers to build and manage real-world AI and ML systems at scale.**

[Metaflow](https://metaflow.org), now supported by [Outerbounds](https://outerbounds.com), streamlines the entire machine learning development lifecycle, from rapid prototyping to reliable production deployments. It enables teams to iterate quickly and deliver robust systems efficiently. Used by a diverse set of companies, including Amazon, Doordash, and Netflix (where it supports thousands of projects), Metaflow unifies code, data, and compute to ensure seamless, end-to-end management of AI/ML systems.

## Key Features

Metaflow offers a comprehensive suite of features to support the entire AI/ML workflow:

*   **Rapid Prototyping & Experimentation:**
    *   Pythonic API for ease of use.
    *   Built-in support for notebooks and rapid local prototyping.
    *   Integrated experiment tracking, versioning, and visualization.
*   **Scalable Compute and Data Management:**
    *   Effortlessly scale workloads using CPUs and GPUs in your cloud environment.
    *   Fast data access for efficient processing.
    *   Support for both embarrassingly parallel and gang-scheduled compute workloads.
    *   Reliable and efficient execution with built-in failure handling and checkpointing.
*   **Production-Ready Deployment:**
    *   Simplified dependency management.
    *   One-click deployment to production-grade workflow orchestrators.
    *   Support for reactive orchestration.

## From Prototype to Production

Metaflow provides a clear path from initial ideas to deployed systems:

![Metaflow Workflow](docs/prototype-to-prod.png)

## Getting Started

### Installation

Install Metaflow via pip:

```bash
pip install metaflow
```

Or using conda:

```bash
conda install -c conda-forge metaflow
```

### Resources

*   **Tutorial:** Get started with the [tutorial](https://docs.metaflow.org/getting-started/tutorials) to create and run your first Metaflow flow.
*   **How Metaflow Works:** Dive deeper into the framework with this [resource](https://docs.metaflow.org/metaflow/basics).
*   **Additional Resources:** Explore additional [Metaflow resources](https://docs.metaflow.org/introduction/metaflow-resources).

### Deploying Infrastructure

To leverage Metaflow's scalability, follow the [guide](https://outerbounds.com/engineering/welcome/) to configure Metaflow and its supporting infrastructure in your cloud environment.

![Multi-Cloud Deployment](docs/multicloud.png)

## Community and Support

*   **Join our Slack community:** [Slack workspace](http://slack.outerbounds.co/) for support and discussions.
*   **API Reference:** Detailed [API Documentation](https://docs.metaflow.org/api).
*   **Release Notes:** Stay updated with the latest features and improvements in our [Release Notes](https://github.com/Netflix/metaflow/releases).

## Contribute

We welcome contributions to Metaflow. Please see our [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow) for more details.

**[View the original Metaflow repository on GitHub](https://github.com/Netflix/metaflow)**