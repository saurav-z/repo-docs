![Metaflow_Logo_Horizontal_FullColor_Ribbon_Dark_RGB](https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png)

# Metaflow: Build, Manage, and Scale Your AI/ML Systems with Ease

**Metaflow, a human-centric framework, empowers data scientists and engineers to efficiently build and manage production-ready AI and ML systems.** Originally developed at Netflix and now supported by Outerbounds, Metaflow streamlines the entire machine learning lifecycle, from rapid prototyping to reliable production deployments, enabling teams to iterate quickly and deliver robust solutions.

[Explore the Metaflow Repository on GitHub](https://github.com/Netflix/metaflow)

## Key Features of Metaflow

*   **Simplified Workflow Management:** Metaflow provides an intuitive Python API that unifies code, data, and compute, simplifying the development cycle from prototyping to production.
*   **Rapid Prototyping & Experimentation:** Quickly prototype locally with built-in support for notebooks, experiment tracking, versioning, and visualization.
*   **Scalable Infrastructure:** Effortlessly scale your workloads horizontally and vertically in the cloud, supporting both CPUs and GPUs for efficient data processing.
*   **Reliable Production Deployments:** Deploy with one-click to highly available production orchestrators with built-in support for reactive orchestration.
*   **Comprehensive Dependency Management:** Easily manage dependencies and ensure consistent environments across your workflow.
*   **Efficient Data Access:** Designed for fast data access to support large-scale, data-intensive workflows.
*   **Wide Adoption:** Powers thousands of AI/ML projects across various companies, including Amazon, Doordash, Dyson, Goldman Sachs, and many more.

## From Prototype to Production and Beyond

Metaflow provides a simple and friendly pythonic API that covers foundational needs of AI and ML systems:
<img src="./docs/prototype-to-prod.png" width="800px">

1.  [Rapid local prototyping](https://docs.metaflow.org/metaflow/basics), [support for notebooks](https://docs.metaflow.org/metaflow/managing-flows/notebook-runs), and built-in support for [experiment tracking, versioning](https://docs.metaflow.org/metaflow/client) and [visualization](https://docs.metaflow.org/metaflow/visualizing-results).
2.  [Effortlessly scale horizontally and vertically in your cloud](https://docs.metaflow.org/scaling/remote-tasks/introduction), utilizing both CPUs and GPUs, with [fast data access](https://docs.metaflow.org/scaling/data) for running [massive embarrassingly parallel](https://docs.metaflow.org/metaflow/basics#foreach) as well as [gang-scheduled](https://docs.metaflow.org/scaling/remote-tasks/distributed-computing) compute workloads [reliably](https://docs.metaflow.org/scaling/failures) and [efficiently](https://docs.metaflow.org/scaling/checkpoint/introduction).
3.  [Easily manage dependencies](https://docs.metaflow.org/scaling/dependencies) and [deploy with one-click](https://docs.metaflow.org/production/introduction) to highly available production orchestrators with built in support for [reactive orchestration](https://docs.metaflow.org/production/event-triggering).

For more details, refer to the [API Reference](https://docs.metaflow.org/api) or the [Release Notes](https://github.com/Netflix/metaflow/releases).

## Getting Started with Metaflow

### Installation

Install Metaflow using pip:

```bash
pip install metaflow
```

or using conda-forge:

```bash
conda install -c conda-forge metaflow
```

Begin with the [tutorial](https://docs.metaflow.org/getting-started/tutorials) to learn the basics of creating and running your first Metaflow flow.

### Additional Resources

*   [How Metaflow Works](https://docs.metaflow.org/metaflow/basics)
*   [Metaflow Resources](https://docs.metaflow.org/introduction/metaflow-resources)

Need help? Join the [Slack community](http://slack.outerbounds.co/)!

### Deploying Infrastructure for Metaflow
<img src="./docs/multicloud.png" width="800px">

To fully leverage Metaflow's scaling and production deployment capabilities, configure your infrastructure by following this [guide](https://outerbounds.com/engineering/welcome/).

## Get Involved

We welcome contributions to Metaflow. See the [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow) for more details.

## Connect with the Community

Join our [Slack workspace](http://slack.outerbounds.co/)!