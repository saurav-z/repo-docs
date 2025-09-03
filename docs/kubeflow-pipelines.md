# Kubeflow Pipelines: Automate and Scale Your Machine Learning Workflows

**Kubeflow Pipelines empower data scientists and ML engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes.** ([Original Repo](https://github.com/kubeflow/pipelines))

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Simplify the process of building and managing complex ML pipelines.
*   **Experimentation:** Easily try out new ideas and techniques, tracking your experiments for optimal results.
*   **Reusability:** Reuse pipeline components and entire pipelines to accelerate development and avoid repetitive work.
*   **Scalability:** Leverage the power of Kubernetes to scale your ML workflows as your needs grow.
*   **Container Runtime Agnostic:** Supports various container runtimes for flexible deployment options.

## Installation

You can install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.

## Dependencies Compatibility

*   **Argo Workflows:** v3.5, v3.6
*   **MySQL:** v8

## Documentation and Resources

*   **Getting Started:** Explore the [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) to begin.
*   **SDK:** Learn how to [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API:** Access the [Pipelines API Documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** Consult the [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) for pipeline development.
*   **AI-Powered Documentation:** Explore the AI-generated documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

## Contributing

Contribute to Kubeflow Pipelines by following the guidelines in the [CONTRIBUTING.md](./CONTRIBUTING.md) file. Build and deploy from source using the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)

*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. Find details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Learn about the KFP Architecture in [Architecture.md](docs/Architecture.md).

## Blog Posts and Articles

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgements

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestration. We appreciate the Argo community's support.