# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows

**Kubeflow Pipelines empowers data scientists and ML engineers to build, deploy, and manage end-to-end machine learning pipelines on Kubernetes, making ML workflows simple, portable, and scalable.** (Original repo: [https://github.com/kubeflow/pipelines](https://github.com/kubeflow/pipelines))

## Key Features of Kubeflow Pipelines

*   **End-to-end orchestration:** Simplify the orchestration of complex machine learning pipelines.
*   **Easy experimentation:** Facilitate rapid prototyping and testing of numerous ML ideas.
*   **Component and Pipeline Reusability:** Quickly assemble end-to-end solutions without rebuilding.
*   **Kubernetes-Native:** Leverage the power and scalability of Kubernetes for ML workflows.
*   **Container Runtime Agnostic:** Run pipelines on any Kubernetes cluster with various container runtimes.

## Installation

Kubeflow Pipelines can be installed in the following ways:

*   **As part of the Kubeflow Platform:** Follow the instructions in the [Kubeflow Platform documentation](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   **As a standalone service:** Deploy Kubeflow Pipelines independently using the guides available [here](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation and Resources

*   **Overview:** Get started with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/) for a comprehensive introduction.
*   **SDK:** Learn to use the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API:** Explore the [Kubeflow Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for detailed API specifications.
*   **SDK Reference:** Refer to the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.
*   **Deep Wiki:** Check out AI-generated documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

## Contributing

Contribute to the project by following the guidelines in [How to Contribute](./CONTRIBUTING.md).
For building and deploying from source, see the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday from 10-11AM (PST). [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x), [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Connect with the community on the `#kubeflow-pipelines` channel within the Cloud Native Computing Foundation Slack workspace. Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels).

## Architecture

For detailed information on the KFP Architecture, please see [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources, and we are grateful for the support of the Argo community.