# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows

**Kubeflow Pipelines** empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, streamlining your ML lifecycle. ([See the original repo](https://github.com/kubeflow/pipelines))

## Key Features:

*   **End-to-End Orchestration:** Simplify the management of complex ML pipelines.
*   **Easy Experimentation:** Effortlessly try out various ML ideas and techniques, managing trials and experiments.
*   **Component & Pipeline Reusability:** Build solutions faster by reusing existing components and pipelines.
*   **Kubernetes Native:** Leverages Kubernetes for portability and scalability.
*   **Flexible Executor:** Supports various container runtimes, including Emissary Executor.

## Getting Started:

### Installation

*   Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.
*   Utilizes Emissary Executor by default for compatibility with various [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/) on Kubernetes.

### Documentation

*   **Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK:** [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community & Resources:

*   **Deep Wiki:** Check out our AI Powered repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).
*   **Contributing:** Review the [Contribution guidelines](./CONTRIBUTING.md).
*   **Developer Guide:** Learn how to build and deploy from source via the [developer guide](./developer_guide.md).
*   **Community Meeting:** Bi-weekly meeting on Wednesdays, 10-11 AM PST.
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace ([https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Architecture:

*   Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts & Resources:

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.