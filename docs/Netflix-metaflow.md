[![Metaflow Logo](https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png)](https://github.com/Netflix/metaflow)

# Metaflow: Build, Manage, and Scale Your AI/ML Systems

Metaflow, originally developed at Netflix, is a human-centric framework designed to empower data scientists and engineers to build and manage robust, production-ready AI and ML systems, streamlining the entire development lifecycle. **[Explore the original Metaflow repository](https://github.com/Netflix/metaflow).**

Metaflow simplifies the complexities of the AI/ML workflow, from rapid prototyping to scalable production deployments, enabling teams of all sizes to iterate quickly and deliver powerful systems efficiently.  It is now supported by [Outerbounds](https://outerbounds.com) and used by companies like Amazon, Doordash, and Netflix.

## Key Features

*   **Rapid Prototyping and Experimentation:**
    *   Develop and test locally with ease.
    *   Seamlessly integrate with notebooks.
    *   Built-in support for experiment tracking, versioning, and visualization.
*   **Scalable and Reliable Execution:**
    *   Horizontally and vertically scale workloads on your cloud (CPUs and GPUs).
    *   Fast data access for efficient processing.
    *   Handles both embarrassingly parallel and gang-scheduled compute workloads.
    *   Resilient design with built-in failure handling.
*   **Production-Ready Deployment:**
    *   Effortless dependency management.
    *   One-click deployment to highly available production orchestrators.
    *   Support for reactive orchestration.

## From Prototype to Production: A Simplified Workflow

Metaflow's user-friendly Python API streamlines the AI/ML lifecycle:

![Metaflow Workflow Diagram](docs/prototype-to-prod.png)

## Getting Started

### Installation

Install Metaflow using pip:

```bash
pip install metaflow
```

Alternatively, using conda-forge:

```bash
conda install -c conda-forge metaflow
```

### Learn More

*   **Tutorial:** Get started quickly with our [tutorial](https://docs.metaflow.org/getting-started/tutorials).
*   **How Metaflow Works:** Understand the fundamentals at [How Metaflow works](https://docs.metaflow.org/metaflow/basics).
*   **Additional Resources:** Explore more features at [Additional resources](https://docs.metaflow.org/introduction/metaflow-resources).
*   **Metaflow Sandbox:** Run and explore Metaflow in seconds with [Metaflow sandbox](https://outerbounds.com/sandbox)

### Deploying Infrastructure

To fully leverage Metaflow's scalability, configure it for your cloud environment using this [guide](https://outerbounds.com/engineering/welcome/).

![Multi-cloud Diagram](docs/multicloud.png)

## Community and Contribution

*   **Join the Community:** Connect with us on our [Slack community](http://slack.outerbounds.co/)!
*   **Contribute:** We welcome contributions! See our [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow) for details.