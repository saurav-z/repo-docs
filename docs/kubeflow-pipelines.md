# Kubeflow Pipelines: Build, Deploy, and Manage Scalable ML Workflows

**Kubeflow Pipelines is the leading open-source solution for orchestrating end-to-end machine learning workflows on Kubernetes.**  ([Original Repo](https://github.com/kubeflow/pipelines))

## Key Features

*   **End-to-End Orchestration:** Simplify the building and management of complex machine learning pipelines.
*   **Easy Experimentation:** Rapidly test new ideas and techniques, and easily manage your trials and experiments.
*   **Component and Pipeline Reusability:** Leverage pre-built components and pipelines for faster development and deployment.
*   **Container Runtime Agnostic:** Run on Kubernetes clusters with any container runtime, including those that have deprecated Docker.

## Overview

Kubeflow Pipelines is a core component of [Kubeflow](https://www.kubeflow.org/), a powerful toolkit for deploying ML workflows on Kubernetes. It empowers you to create reusable, scalable, and portable machine learning pipelines using the Kubeflow Pipelines SDK.

## Installation

Kubeflow Pipelines can be installed:

*   As part of the complete [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   As a standalone service via the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation

*   **Getting Started:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Documentation:** [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-Powered Documentation:** [DeepWiki](https://deepwiki.com/kubeflow/pipelines)

## Contributing

Read the [How to Contribute](./CONTRIBUTING.md) guidelines. Learn to build and deploy Kubeflow Pipelines from source in the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Every other Wed 10-11AM (PST)
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  #kubeflow-pipelines on the Cloud Native Computing Foundation Slack workspace ([details](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Architecture

*   [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines relies on [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources. We are grateful for their support.