# Kubeflow Pipelines: Build, Deploy, and Manage End-to-End Machine Learning Workflows

**Kubeflow Pipelines** empowers data scientists and engineers to build, deploy, and manage robust, scalable machine learning workflows on Kubernetes.  Check out the [original repository](https://github.com/kubeflow/pipelines) for the source code and more details.

## Key Features

*   **End-to-End Orchestration:** Simplify the orchestration of complex ML pipelines from start to finish.
*   **Experimentation Made Easy:**  Streamline your experimentation process, enabling you to easily try new ideas and techniques and manage various trials/experiments.
*   **Component and Pipeline Reusability:** Quickly assemble end-to-end solutions with reusable components and pipelines, minimizing repetitive build efforts.
*   **Container Runtime Agnostic:** Runs on any Kubernetes cluster.

## Overview

[Kubeflow](https://www.kubeflow.org/) is a powerful machine learning (ML) toolkit designed to simplify, enhance portability, and scale deployments of ML workflows on Kubernetes. **Kubeflow Pipelines** leverages the Kubeflow Pipelines SDK to create reusable, end-to-end ML workflows.  This service helps to achieve:

*   **Orchestration:** Simplifying the process of managing ML pipelines.
*   **Experimentation:**  Providing a straightforward way to experiment with different ML approaches.
*   **Reusability:**  Enabling the reuse of components and pipelines for rapid solution development.

## Installation

*   Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy Kubeflow Pipelines as a standalone service using the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).
*   Uses [Emissary Executor](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/choose-executor/#emissary-executor) by default.

## Documentation & Resources

*   **Getting Started:**  Explore your first pipeline and dive deeper with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Usage:**  Discover various methods to [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:**  Access the Kubeflow [Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for comprehensive API specifications.
*   **Python SDK:** Consult the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) for writing pipelines using the Python SDK.
*   **AI-Powered Documentation:** Explore the AI-generated documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).
    > :warning: *Please note, this is AI generated and may not have completely accurate information.*

## Contributing

Contribute to Kubeflow Pipelines by reviewing the guidelines in [How to Contribute](./CONTRIBUTING.md). To learn how to build and deploy Kubeflow Pipelines from source code, refer to the [developer guide](./developer_guide.md).

## Community

### Community Meeting

*   Occurs every other Wednesday from 10-11 AM (PST).
*   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
*   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
*   [Meeting Notes](http://bit.ly/kfp-meeting-notes)

### Slack

*   Join the Kubeflow Pipelines channel (#kubeflow-pipelines) on the Cloud Native Computing Foundation Slack workspace. Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels).

## Architecture

*   Learn more about the KFP Architecture at [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default for orchestrating Kubernetes resources. The Argo community has been very supportive, and we are grateful for their contribution.