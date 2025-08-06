# Kubeflow Pipelines: Orchestrate and Automate Your Machine Learning Workflows

**Kubeflow Pipelines** empowers you to build, deploy, and manage reusable end-to-end machine learning (ML) workflows. Learn more about this powerful tool on the [Kubeflow Pipelines GitHub repository](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-End Orchestration:** Simplify the management of complex ML pipelines.
*   **Experimentation:**  Easily test and compare various ML ideas and techniques.
*   **Reusability:** Build and reuse components and pipelines for faster development.
*   **Container Runtime Agnostic:** Runs on Kubernetes clusters with any container runtime thanks to the Emissary Executor.

## Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.

## Documentation

*   **Getting Started:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK:**  [Use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Deep Wiki

Explore AI-powered documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning: Please note, this is AI generated and may not have completely accurate information.

## Contributing

Contribute to Kubeflow Pipelines by following the guidelines in [How to Contribute](./CONTRIBUTING.md) and the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Every other Wed 10-11AM (PST). [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x), [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  Find us in the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. More details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Learn about the KFP Architecture at [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.