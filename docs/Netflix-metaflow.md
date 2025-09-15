<!-- SEO-optimized README for Metaflow -->
![Metaflow Logo](https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png)

# Metaflow: Build, Manage, and Scale Your AI/ML Systems

**Metaflow is a human-centric framework designed to streamline the entire AI and ML development lifecycle, from prototyping to production.**  Originally developed at Netflix and now supported by Outerbounds, Metaflow empowers data scientists and engineers to build and manage real-world AI and ML systems efficiently.  [Explore the original repository](https://github.com/Netflix/metaflow) for the source code.

## Key Features of Metaflow:

Metaflow simplifies the complexities of building and deploying AI/ML projects with these powerful capabilities:

*   **Rapid Prototyping & Experimentation:**  Quickly prototype in notebooks, track experiments, and visualize results.
*   **Scalable Compute:** Easily scale workloads horizontally and vertically in the cloud using CPUs and GPUs.
*   **Reliable & Efficient Data Access:**  Run massive, embarrassingly parallel compute tasks with fast data access.
*   **Dependency Management:** Effortlessly manage dependencies for reproducible results.
*   **Production-Ready Deployment:** Deploy your models with one-click to highly available production orchestrators.
*   **Reactive Orchestration:** Built-in support for event-triggered workflows.
*   **Versioning & Reproducibility:** Built-in experiment tracking, versioning and visualization capabilities for robust workflows.

## From Prototype to Production

Metaflow provides a simple and friendly Pythonic API that covers the foundational needs of AI and ML systems:

[Image of prototype to production flow - original file](docs/prototype-to-prod.png)

1.  **Rapid local prototyping, support for notebooks, and built-in support for experiment tracking, versioning and visualization.**
2.  **Effortlessly scale horizontally and vertically in your cloud, utilizing both CPUs and GPUs, with fast data access for running massive embarrassingly parallel as well as gang-scheduled compute workloads reliably and efficiently.**
3.  **Easily manage dependencies and deploy with one-click to highly available production orchestrators with built in support for reactive orchestration.**

## Getting Started with Metaflow

Getting started with Metaflow is easy:

### Installation

Install Metaflow using `pip`:

```bash
pip install metaflow
```

Or using `conda`:

```bash
conda install -c conda-forge metaflow
```

### Tutorials & Resources

*   **Tutorial:**  Follow our [tutorial](https://docs.metaflow.org/getting-started/tutorials) to create and run your first Metaflow flow.
*   **How Metaflow Works:**  [How Metaflow Works](https://docs.metaflow.org/metaflow/basics)
*   **Additional Resources:**  [Metaflow Resources](https://docs.metaflow.org/introduction/metaflow-resources)
*   **Metaflow Sandbox:** Explore in seconds with [Metaflow sandbox](https://outerbounds.com/sandbox)

## Deploying Infrastructure in Your Cloud

[Image of Multi-Cloud deployment flow - original file](docs/multicloud.png)

To leverage Metaflow's scaling and production features, configure your infrastructure using this [guide](https://outerbounds.com/engineering/welcome/).

## Get Involved

*   **Community:** Join our [Slack community](http://slack.outerbounds.co/) to connect with other users and the Metaflow team.
*   **Contributing:**  We welcome contributions!  See our [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow) for details.