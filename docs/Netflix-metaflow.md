<!--
  SPDX-License-Identifier: Apache-2.0
-->
<img src="https://user-images.githubusercontent.com/763451/89453116-96a57e00-d713-11ea-9fa6-82b29d4d6eff.png" alt="Metaflow Logo" width="300">

# Metaflow: Build and Manage AI/ML Systems with Ease

**Metaflow is a human-centric framework designed to empower data scientists and engineers to build, manage, and deploy robust AI and ML systems efficiently.** Originally developed at Netflix and now supported by Outerbounds, Metaflow simplifies the entire machine learning lifecycle, from rapid prototyping to production deployment.  

**[Explore the Metaflow Repository on GitHub](https://github.com/Netflix/metaflow)**

## Key Features & Benefits

Metaflow provides a streamlined experience for building and deploying AI/ML systems, offering:

*   **Rapid Prototyping:**  Quickly experiment and iterate in notebooks with built-in support for experiment tracking, versioning, and visualization.
*   **Scalable Compute:** Easily scale workloads horizontally and vertically in your cloud environment, leveraging CPUs and GPUs for both embarrassingly parallel and gang-scheduled tasks.
*   **Production-Ready Deployment:** Deploy with one-click to highly available production orchestrators with reactive orchestration capabilities.
*   **Simplified Dependency Management:** Effortlessly manage dependencies across your projects, ensuring reproducibility and consistency.
*   **End-to-End Management:** Unifies code, data, and compute for seamless management of AI/ML systems from development to production.

## From Prototype to Production: A Unified Workflow

<img src="./docs/prototype-to-prod.png" width="800px" alt="Metaflow Workflow">

Metaflow offers a Pythonic API designed to simplify the complexities of the AI/ML lifecycle:

1.  **Prototype & Experiment:** Accelerate development with rapid prototyping, notebook support, and built-in tools for tracking, versioning, and visualization.
2.  **Scale & Optimize:** Effortlessly scale your workflows across various cloud environments, utilizing CPUs and GPUs for both parallel and distributed compute workloads.
3.  **Deploy & Monitor:**  One-click deployment to production-grade orchestrators with reactive orchestration.

## Getting Started with Metaflow

Metaflow is easy to install and use. The [Metaflow sandbox](https://outerbounds.com/sandbox) allows you to quickly get started.

### Installation

Install Metaflow using `pip`:

```bash
pip install metaflow
```

Or, with `conda`:

```bash
conda install -c conda-forge metaflow
```

### Resources

*   **Tutorial:** Follow our [tutorial](https://docs.metaflow.org/getting-started/tutorials) to create and run your first Metaflow flow.
*   **How Metaflow Works:** Learn the fundamentals at [How Metaflow works](https://docs.metaflow.org/metaflow/basics).
*   **Additional Resources:**  Explore additional resources at [Additional resources](https://docs.metaflow.org/introduction/metaflow-resources).

## Deploying Infrastructure

<img src="./docs/multicloud.png" width="800px" alt="Multi-cloud Deployment">

To fully leverage Metaflow's capabilities for scaling and production deployment, configure your infrastructure following this [guide](https://outerbounds.com/engineering/welcome/).

## Community & Contribution

*   **Join our Community:**  Connect with us and other users in our [Slack workspace](http://slack.outerbounds.co/).
*   **Contribute:** We welcome contributions!  Review our [contribution guide](https://docs.metaflow.org/introduction/contributing-to-metaflow).