# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Kubeflow Pipelines empower you to create, automate, and manage end-to-end machine learning workflows on Kubernetes, making ML deployments simple, portable, and scalable.**  [Explore the Kubeflow Pipelines repository](https://github.com/kubeflow/pipelines)

Key Features:

*   **End-to-End Orchestration:** Simplify the management of your entire machine learning pipeline.
*   **Experimentation:** Easily iterate and test different ideas and techniques with streamlined trials and experiments.
*   **Reusability:**  Leverage pre-built components and pipelines for faster development and deployment.
*   **Scalability**: Built to run on Kubernetes, KFP allows you to scale your pipelines up or down based on resource needs.
*   **Container Runtime Agnostic**:  Kubeflow Pipelines can run on any Kubernetes cluster regardless of the underlying container runtime.

## Getting Started

*   **Installation:**  Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a [standalone service](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

*   **Documentation:**
    *   **Kubeflow Pipelines Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
    *   **Kubeflow Pipelines SDK:** [Use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
    *   **Kubeflow Pipelines API:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
    *   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community and Contributing

*   **Contribute:** Review the [How to Contribute](./CONTRIBUTING.md) guidelines.
*   **Developer Guide:** Learn to build and deploy from source with the [developer guide](./developer_guide.md).
*   **Community Meeting:** Every other Wednesday, 10-11 AM PST ([Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Meeting notes](http://bit.ly/kfp-meeting-notes)).
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace ([Kubeflow Slack Channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)).

## Architecture

*   **Architecture:** Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts & Resources

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.