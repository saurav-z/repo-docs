# Kubeflow Pipelines: Orchestrate, Experiment, and Reuse Your ML Workflows

**Kubeflow Pipelines** empowers machine learning engineers to build, deploy, and manage end-to-end ML workflows on Kubernetes, making deployments simple, portable, and scalable.  For more details, visit the [original Kubeflow Pipelines repository](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-End Orchestration:** Simplify the orchestration of your complete machine learning pipelines.
*   **Easy Experimentation:**  Streamline the process of trying new ideas, techniques, and managing your experiments.
*   **Component and Pipeline Reusability:** Quickly assemble end-to-end solutions by reusing components and pipelines.
*   **Container Runtime Agnostic:**  Run Kubeflow Pipelines on Kubernetes clusters with any container runtime.

## Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.

## Documentation

*   **Overview:** [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:** [Use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Documentation:** [Kubeflow Pipelines API Doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **DeepWiki:** Check out our AI Powered repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).
    > :warning: Please note, this is AI generated and may not have completely accurate information.

## Contributing & Community

*   **Contribution Guidelines:**  [How to Contribute](./CONTRIBUTING.md)
*   **Developer Guide:**  [developer_guide.md](./developer_guide.md)
*   **Community Meeting:** Every other Wednesday, 10-11 AM PST ([Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x), [Meeting notes](http://bit.ly/kfp-meeting-notes))
*   **Slack:**  #kubeflow-pipelines on the Cloud Native Computing Foundation Slack workspace ([https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Architecture

*   [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.