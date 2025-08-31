# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning (ML) workflows on Kubernetes, making your ML lifecycle simple, portable, and scalable.** ([View the original repo](https://github.com/kubeflow/pipelines))

## Key Features

*   **End-to-End Orchestration:** Simplify and automate your ML pipeline orchestration.
*   **Experimentation:** Easily explore different ideas and techniques with built-in experiment management.
*   **Reusability:** Build upon existing components and pipelines for rapid development and deployment.
*   **Kubernetes Native:** Designed to run seamlessly on Kubernetes, leveraging its scalability and portability.
*   **Component Library:** Offers a rich set of pre-built components and integrations.
*   **Open Source:** Benefit from an active community and open-source development.

## Overview

Kubeflow Pipelines is a crucial component of the [Kubeflow](https://www.kubeflow.org/) ecosystem, a dedicated toolkit for simplifying the deployment of ML workflows on Kubernetes. Kubeflow Pipelines enables the creation of reusable end-to-end ML workflows using the Kubeflow Pipelines SDK.

## Installation

Kubeflow Pipelines offers flexible installation options:

*   **Kubeflow Platform:** Install as part of the comprehensive [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   **Standalone Service:** Deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) as a standalone service.

**Important Note:** From Kubeflow Pipelines 1.8, Emissary Executor is the default executor.  It's container runtime agnostic.

### Dependencies Compatibility Matrix

| Dependency     | Versions    |
| -------------- | ----------  |
| Argo Workflows | v3.5, v3.6  |
| MySQL          | v8          |

## Documentation & Resources

*   **Getting Started:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:** [Kubeflow Pipelines SDK documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace.  Find details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Deep Wiki (AI-Generated)

Explore AI-powered documentation for this project at [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning:  Please note that this is AI-generated and may contain inaccuracies.

## Contributing

We welcome contributions!  Please review the [How to Contribute](./CONTRIBUTING.md) guidelines.  For building and deploying Kubeflow Pipelines from source, see the [developer guide](./developer_guide.md).

## Architecture

Detailed information on the Kubeflow Pipelines architecture can be found in [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows). We are grateful to the Argo community for their support.