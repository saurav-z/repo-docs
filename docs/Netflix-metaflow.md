![Metaflow_Logo_Horizontal_FullColor_Ribbon_Dark_RGB](https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png)

# Metaflow: Build, Manage, and Scale AI/ML Systems

**Metaflow, originally developed at Netflix, is a human-centric framework empowering data scientists and engineers to efficiently build and manage real-world AI and ML systems, from prototyping to production.**

[View the original Metaflow repository on GitHub](https://github.com/Netflix/metaflow)

Metaflow streamlines the entire AI/ML development lifecycle, serving teams of all sizes and enabling rapid iteration and reliable deployment.  It unifies code, data, and compute, ensuring seamless end-to-end management of your AI/ML projects.  Supported by [Outerbounds](https://outerbounds.com), Metaflow boosts productivity for research and engineering teams working on projects from classical statistics to cutting-edge deep learning and foundation models. It's used by diverse companies like Amazon, Doordash, Dyson, Goldman Sachs, and many others, managing petabytes of data and supporting thousands of AI/ML projects.

## Key Features of Metaflow

*   **Rapid Prototyping & Experimentation:**
    *   Quickly prototype with Python and notebook support.
    *   Built-in experiment tracking, versioning, and visualization.
*   **Scalable Compute & Data Handling:**
    *   Seamlessly scale workloads horizontally and vertically using CPUs and GPUs.
    *   Fast data access for efficient processing of massive datasets.
    *   Reliable and efficient execution of both parallel and gang-scheduled compute workloads.
*   **Simplified Deployment & Management:**
    *   Easy dependency management.
    *   One-click deployment to production-grade workflow orchestrators.
    *   Support for reactive orchestration.

## From Prototype to Production and Back

Metaflow provides a straightforward Pythonic API that covers the essential needs of AI/ML systems:

<img src="./docs/prototype-to-prod.png" width="800px">

## Getting Started with Metaflow

Get up and running quickly with Metaflow!

### Installation

Install Metaflow using pip:

```bash
pip install metaflow
```

Alternatively, install using conda-forge:

```bash
conda install -c conda-forge metaflow
```

Start by exploring the [Metaflow sandbox](https://outerbounds.com/sandbox) or follow the [tutorial](https://docs.metaflow.org/getting-started/tutorials) to create and run your first Metaflow flow.

### Additional Resources:

*   [How Metaflow Works](https://docs.metaflow.org/metaflow/basics)
*   [Additional Resources](https://docs.metaflow.org/introduction/metaflow-resources)
*   [API Reference](https://docs.metaflow.org/api)
*   [Release Notes](https://github.com/Netflix/metaflow/releases)

## Deploying Infrastructure in Your Cloud

<img src="./docs/multicloud.png" width="800px">

To leverage the full power of Metaflow, which includes scaling to external compute clusters and deploying to production-grade orchestrators, follow this [guide](https://outerbounds.com/engineering/welcome/) to configure Metaflow and its supporting infrastructure.

## Get Involved

*   **Join our Slack community:** [Slack workspace](http://slack.outerbounds.co/)
*   **Contribute:** See our [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow).