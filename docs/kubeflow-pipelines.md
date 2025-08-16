# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Simplify and automate your machine learning lifecycle with Kubeflow Pipelines, a powerful toolkit for building and deploying scalable ML workflows on Kubernetes.** [Learn more on the original repository](https://github.com/kubeflow/pipelines).

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Easily orchestrate complex ML pipelines from data ingestion to model deployment.
*   **Experimentation Made Simple:** Streamline the process of experimenting with different ML techniques and track your trials efficiently.
*   **Component and Pipeline Reusability:** Build upon existing components and pipelines to accelerate the development of end-to-end ML solutions.
*   **Container Runtime Agnostic:** Supports various container runtimes for flexible deployment.

## Installation

*   Integrate Kubeflow Pipelines seamlessly within the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy Kubeflow Pipelines as a standalone service using the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation & Resources

*   **Overview:** Get started with Kubeflow Pipelines and explore its capabilities with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Learn how to leverage the Kubeflow Pipelines SDK with the [Kubeflow Pipelines SDK documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Dive into the [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for detailed API specifications.
*   **Python SDK Reference:** Consult the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.
*   **AI-Powered Documentation:** Explore AI-generated documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).
    >   :warning: Please note, this is AI generated and may not have completely accurate information.

## Contributing

Contribute to the development of Kubeflow Pipelines by following the guidelines outlined in [How to Contribute](./CONTRIBUTING.md). For building and deploying from source, consult the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday at 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVaZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Connect with the community on the [#kubeflow-pipelines](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) channel on the Cloud Native Computing Foundation Slack workspace.

## Architecture

*   For detailed information on the KFP architecture, refer to [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines relies on [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources. The project is grateful for the Argo community's support.