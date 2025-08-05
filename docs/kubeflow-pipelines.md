# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Streamline your machine learning lifecycle with Kubeflow Pipelines, enabling reusable, end-to-end ML workflows on Kubernetes.** [Explore the Kubeflow Pipelines Repository](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-End Orchestration:** Simplify and manage the complete machine learning pipeline process.
*   **Experimentation Made Easy:** Quickly test, iterate, and manage multiple ML trials and experiments.
*   **Component and Pipeline Reusability:**  Build solutions faster by reusing existing components and pipelines.
*   **Container Runtime Agnostic**: Run Kubeflow Pipelines on any Kubernetes cluster with various container runtimes.

## Overview

Kubeflow Pipelines, built on the [Kubeflow](https://www.kubeflow.org/) toolkit, provides a powerful solution for building, deploying, and managing machine learning workflows on Kubernetes. It allows data scientists and ML engineers to create reusable pipelines for tasks such as data preprocessing, model training, evaluation, and deployment, improving efficiency and reproducibility in the ML lifecycle.

## Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service. The Emissary Executor is used by default, making it compatible with all [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/).

## Documentation & Resources

*   **Kubeflow Pipelines Overview:** ([https://www.kubeflow.org/docs/components/pipelines/overview/](https://www.kubeflow.org/docs/components/pipelines/overview/))
*   **Kubeflow Pipelines SDK Documentation:** ([https://kubeflow-pipelines.readthedocs.io/en/stable/](https://kubeflow-pipelines.readthedocs.io/en/stable/))
*   **Kubeflow Pipelines API Documentation:** ([https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/))
*   **Python SDK Reference:** ([https://kubeflow-pipelines.readthedocs.io/en/stable/](https://kubeflow-pipelines.readthedocs.io/en/stable/))
*   **Deep Wiki (AI-Powered Documentation):** ([https://deepwiki.com/kubeflow/pipelines](https://deepwiki.com/kubeflow/pipelines))

## Contributing

Contribute to the project by reading the guidelines in [How to Contribute](./CONTRIBUTING.md) and the [developer guide](./developer_guide.md).

## Community

*   **Community Meetings:** Occur bi-weekly on Wednesdays (10-11 AM PST). [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL). [Meeting notes](http://bit.ly/kfp-meeting-notes).
*   **Slack:**  Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace.  More details:  [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

*   Detailed architecture information can be found in [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestration. The project is grateful for the Argo community's support.