# Kubeflow Pipelines: Build, Deploy, and Manage Your ML Workflows

**Orchestrate your machine learning pipelines with ease using Kubeflow Pipelines, a powerful toolkit built for Kubernetes.** [(Original Repository)](https://github.com/kubeflow/pipelines)

## Key Features

*   **End-to-End Orchestration:** Simplify the management of your entire machine learning lifecycle, from data ingestion to model deployment.
*   **Experimentation Made Easy:** Quickly test and iterate on different ideas and techniques with streamlined experiment tracking and management.
*   **Component and Pipeline Reusability:** Build upon existing components and pipelines, saving time and effort by eliminating the need to rebuild from scratch.
*   **Kubernetes Native:** Seamlessly integrates with Kubernetes for scalable and portable ML workflows.
*   **SDK Support:** Develop and deploy pipelines using the Kubeflow Pipelines SDK, allowing for flexibility in your development.

## Overview

[Kubeflow](https://www.kubeflow.org/) is a leading machine learning (ML) toolkit designed to make ML deployments on Kubernetes simple, portable, and scalable. **Kubeflow Pipelines** are reusable, end-to-end ML workflows, built using the Kubeflow Pipelines SDK.

## Installation

*   Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or deploy it as a standalone service.
*   Kubeflow Pipelines now uses the [Emissary Executor](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/choose-executor/#emissary-executor) by default, providing container runtime agnosticism.

## Documentation

*   [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   [Pipelines API Doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM PST.
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting Notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** #kubeflow-pipelines on the Cloud Native Computing Foundation Slack workspace. ([More Info](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Contributing

Read the guidelines in [How to Contribute](./CONTRIBUTING.md)

For building and deploying Kubeflow Pipelines from source, see the [developer guide](./developer_guide.md).

## Architecture

Details on the KFP architecture can be found in [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgements

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.