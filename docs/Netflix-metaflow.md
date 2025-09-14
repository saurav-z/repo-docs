![Metaflow Logo](https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png)

# Metaflow: Build, Manage, and Scale Your AI & ML Systems with Ease

[Metaflow](https://metaflow.org) is a powerful, human-centric framework designed to simplify the development lifecycle for AI and ML projects, from initial prototyping to robust production deployments. Originally developed at Netflix and now supported by Outerbounds, Metaflow empowers data scientists and engineers to build and manage real-world AI/ML systems efficiently.  This README provides a brief overview of Metaflow; for more detailed information, visit the [original repository](https://github.com/Netflix/metaflow).

## Key Features and Benefits

Metaflow offers a comprehensive suite of features to streamline your AI/ML workflow:

*   **Rapid Prototyping:** Quickly experiment and iterate with Python-based workflows, supported by notebooks and built-in experiment tracking.
*   **Scalable Compute:** Seamlessly scale your workloads across various cloud platforms using CPUs and GPUs, with fast data access for efficient processing.
*   **Simplified Deployment:** Deploy your models to production with one-click deployment to highly available orchestrators, including reactive orchestration.
*   **Version Control & Tracking:** Built-in support for experiment tracking, versioning, and visualization ensures you can track and manage your projects.
*   **Dependency Management:** Easily manage and resolve dependencies within your workflows.

## From Prototype to Production and Back

Metaflow provides a user-friendly Pythonic API that covers foundational needs of AI and ML systems:
<img src="./docs/prototype-to-prod.png" width="800px">

1.  [Rapid local prototyping](https://docs.metaflow.org/metaflow/basics), [support for notebooks](https://docs.metaflow.org/metaflow/managing-flows/notebook-runs), and built-in support for [experiment tracking, versioning](https://docs.metaflow.org/metaflow/client) and [visualization](https://docs.metaflow.org/metaflow/visualizing-results).
2.  [Effortlessly scale horizontally and vertically in your cloud](https://docs.metaflow.org/scaling/remote-tasks/introduction), utilizing both CPUs and GPUs, with [fast data access](https://docs.metaflow.org/scaling/data) for running [massive embarrassingly parallel](https://docs.metaflow.org/metaflow/basics#foreach) as well as [gang-scheduled](https://docs.metaflow.org/scaling/remote-tasks/distributed-computing) compute workloads [reliably](https://docs.metaflow.org/scaling/failures) and [efficiently](https://docs.metaflow.org/scaling/checkpoint/introduction).
3.  [Easily manage dependencies](https://docs.metaflow.org/scaling/dependencies) and [deploy with one-click](https://docs.metaflow.org/production/introduction) to highly available production orchestrators with built in support for [reactive orchestration](https://docs.metaflow.org/production/event-triggering).

## Getting Started

Get up and running quickly with Metaflow:

### Installation

Install Metaflow using `pip`:

```bash
pip install metaflow
```

or using `conda`:

```bash
conda install -c conda-forge metaflow
```

### Explore Metaflow

*   **Tutorial:**  Follow the [tutorial](https://docs.metaflow.org/getting-started/tutorials) to create and run your first Metaflow flow.
*   **Documentation:** Explore comprehensive documentation covering [How Metaflow works](https://docs.metaflow.org/metaflow/basics) and find [Additional resources](https://docs.metaflow.org/introduction/metaflow-resources).
*   **Sandbox:** Explore Metaflow's capabilities with the [Metaflow sandbox](https://outerbounds.com/sandbox)

### Deploying Infrastructure

<img src="./docs/multicloud.png" width="800px">

For scaling and production deployment, follow the [guide](https://outerbounds.com/engineering/welcome/) to configure Metaflow for your cloud infrastructure.

## Get in Touch

Join the Metaflow community!  Connect with us on our [Slack workspace](http://slack.outerbounds.co/) to ask questions, share your experiences, and get help.

## Contribute

We welcome contributions to Metaflow.  See the [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow) for details on how to contribute.