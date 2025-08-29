# Kubeflow Pipelines: Build, Deploy, and Manage End-to-End ML Workflows

**Kubeflow Pipelines empowers data scientists and machine learning engineers to build, deploy, and manage scalable and reproducible machine learning workflows on Kubernetes.** Find the original repo [here](https://github.com/kubeflow/pipelines).

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Simplify the orchestration of your entire machine learning pipeline, from data preparation to model deployment.
*   **Easy Experimentation:** Quickly test and iterate on different ML ideas and techniques, streamlining your experimentation process.
*   **Component and Pipeline Reusability:** Build reusable components and pipelines to accelerate development and avoid redundant work.
*   **Scalability:** Leverage the power of Kubernetes to scale your ML pipelines to handle large datasets and complex models.
*   **Container Runtime Agnostic:** Run Kubeflow Pipelines on Kubernetes clusters with any container runtimes, thanks to the Emissary Executor.

## Getting Started

*   **Installation:** Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service via the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).
*   **Dependencies:**

    | Dependency     | Versions    |
    | -------------- | ----------  |
    | Argo Workflows | v3.5, v3.6  |
    | MySQL          | v8          |

## Documentation and Resources

*   **Overview:** Get started with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Learn how to use the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Explore the [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specifications.
*   **Python SDK:** Refer to the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.
*   **Deep Wiki:** AI-powered repo documentation is available on [DeepWiki](https://deepwiki.com/kubeflow/pipelines). (Note: information may not be entirely accurate.)

## Contributing

Contribute to Kubeflow Pipelines! See [How to Contribute](./CONTRIBUTING.md). and the [developer guide](./developer_guide.md).

## Community

*   **Community Meetings:** Bi-weekly meetings every other Wed 10-11AM (PST)
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace.  More details are available at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

For details about the KFP architecture, see [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows).