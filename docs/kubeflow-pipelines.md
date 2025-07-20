# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Streamline your machine learning lifecycle with Kubeflow Pipelines, a powerful tool for building, deploying, and managing end-to-end ML workflows on Kubernetes.**  Find the original repository [here](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-End Orchestration:** Simplify the creation and management of complex machine learning pipelines.
*   **Experimentation:** Easily test and iterate on different ideas and techniques with robust experiment tracking.
*   **Component and Pipeline Reusability:** Accelerate development by reusing components and pipelines to create solutions quickly.
*   **Container Runtime Agnostic**: Run Kubeflow Pipelines on Kubernetes cluster with any [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/).

## Overview

Kubeflow Pipelines is a core component of [Kubeflow](https://www.kubeflow.org/), an open-source toolkit dedicated to simplifying machine learning deployments on Kubernetes. Kubeflow Pipelines enable you to build reusable ML workflows using the Kubeflow Pipelines SDK.

## Installation

You can install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.

## Documentation & Resources

*   **Getting Started:** [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:** [Kubeflow Pipelines SDK Documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API Documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-Powered Documentation:** Explore AI-generated documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines). (Please note that this is AI-generated and may contain inaccuracies.)
*   **Architecture:** Dive into the technical details in the [Architecture.md](docs/Architecture.md) file.

## Contributing & Community

*   **Contribute:** Review the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.
*   **Developer Guide:** Learn how to build and deploy from source using the [developer\_guide.md](./developer_guide.md).
*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday from 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVaG9vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting Notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the Kubeflow Pipelines Slack channel (#kubeflow-pipelines) on the Cloud Native Computing Foundation Slack workspace. Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Blog Posts

*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) under the hood for orchestrating Kubernetes resources. The Argo community's support is greatly appreciated.