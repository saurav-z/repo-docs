# Kubeflow Pipelines: Automate and Orchestrate Your Machine Learning Workflows

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, streamlining your ML lifecycle.**  Explore the [Kubeflow Pipelines project](https://github.com/kubeflow/pipelines) for more information.

## Key Features of Kubeflow Pipelines:

*   **End-to-End Orchestration:** Simplify the orchestration of complex ML pipelines, from data preparation to model deployment.
*   **Experimentation Made Easy:**  Effortlessly experiment with different ideas and techniques, managing your trials and experiments effectively.
*   **Component and Pipeline Reusability:** Quickly assemble end-to-end solutions by reusing components and pipelines, saving time and effort.
*   **Container Runtime Agnostic:** Runs on Kubernetes clusters with any container runtime due to the Emissary Executor.

## Getting Started with Kubeflow Pipelines:

*   **Installation:** Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.
*   **Documentation:**
    *   Comprehensive [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
    *   Explore the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/) for building pipelines.
    *   Refer to the [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specification.
    *   Use the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) for guidance.

## Community & Resources:

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday.
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Connect with the community on the Kubeflow Pipelines channel (#kubeflow-pipelines) on the Cloud Native Computing Foundation Slack workspace.  More details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels).
*   **Deep Wiki:** Explore AI-powered documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines). *Note: This information may not be completely accurate.*
*   **Contributing:**  Review the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines before contributing. Learn how to build and deploy Kubeflow Pipelines from source in the [developer guide](./developer_guide.md).
*   **Architecture:**  Find details about the KFP Architecture at [Architecture.md](docs/Architecture.md)

## Blog Posts:

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments:

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources, and we're grateful for the Argo community's support.