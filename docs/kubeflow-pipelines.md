# Kubeflow Pipelines: Simplify and Automate Your Machine Learning Workflows

**Kubeflow Pipelines is the go-to solution for building, deploying, and managing end-to-end machine learning workflows on Kubernetes.**  [Explore the original repository](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-End Orchestration:** Streamline your ML workflow by orchestrating complex pipelines from start to finish.
*   **Simplified Experimentation:** Easily test and iterate on various ML ideas and techniques, managing your trials efficiently.
*   **Component & Pipeline Reusability:** Build solutions faster by reusing components and pipelines without rebuilding from scratch.
*   **Scalability:** Leverage the power of Kubernetes to scale your ML workflows to meet your growing needs.
*   **Container Runtime Agnostic:** Runs on Kubernetes clusters with any container runtimes, enabling flexibility in deployment.

## Overview

Kubeflow Pipelines is a core component of Kubeflow, a comprehensive toolkit designed to simplify and accelerate the deployment of ML workflows on Kubernetes. It allows you to define, run, and track complex ML pipelines, enabling faster experimentation, improved collaboration, and increased reproducibility.

## Installation

You can install Kubeflow Pipelines in the following ways:

*   As part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   As a standalone service; detailed instructions are available in the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation and Resources

*   **Kubeflow Pipelines Overview:** Get started with your first pipeline and learn more in the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **Kubeflow Pipelines SDK:** Learn how to use the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Documentation:** Explore the [Kubeflow Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specification.
*   **Python SDK Reference:** Consult the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.
*   **Deep Wiki:** Explore AI-generated documentation via [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

## Contributing

We welcome contributions! Please review the [contributing guidelines](./CONTRIBUTING.md) before getting started.  You can also consult the [developer guide](./developer_guide.md) to learn how to build and deploy Kubeflow Pipelines from source code.

## Community

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday 10-11AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Connect with the community on the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. Find details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Learn more about the Kubeflow Pipelines architecture in [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) under the hood for orchestrating Kubernetes resources. We are grateful for the Argo community's support.