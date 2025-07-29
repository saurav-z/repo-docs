# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, simplifying the ML lifecycle.**  [Get started with Kubeflow Pipelines](https://github.com/kubeflow/pipelines)

Kubeflow Pipelines is a key component of the Kubeflow project, providing a platform for building, deploying, and managing portable and scalable machine learning workflows.

## Key Features

*   **End-to-End Orchestration:** Simplify the orchestration of your entire machine learning pipeline.
*   **Experimentation Made Easy:** Easily try numerous ideas, techniques, and manage your trials and experiments.
*   **Component and Pipeline Reusability:** Build upon and re-use components to accelerate the development of end-to-end solutions.
*   **Container Runtime Agnostic:** Run pipelines on Kubernetes clusters with any container runtime using the Emissary Executor.

## Installation

Kubeflow Pipelines can be installed:

*   As part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   As a standalone service, following the guide [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation

*   **Overview:** Get started with your first pipeline with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Learn how to [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Consult the [Kubeflow Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specifications.
*   **Python SDK Reference:** Explore the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines.
*   **DeepWiki:** Explore AI-generated repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

## Contributing

For guidelines on contributing, refer to [How to Contribute](./CONTRIBUTING.md). Learn how to build and deploy from source in the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Bi-weekly meetings every other Wednesday, 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestration. The Argo community's support is greatly appreciated.