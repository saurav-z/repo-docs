# Kubeflow Pipelines: Streamline Your Machine Learning Workflows

**Kubeflow Pipelines empowers data scientists and ML engineers to build, deploy, and manage end-to-end machine learning pipelines on Kubernetes, simplifying and accelerating the ML lifecycle.**  Discover the power of automated ML workflows.  [Explore the original repository](https://github.com/kubeflow/pipelines).

## Key Features of Kubeflow Pipelines:

*   **End-to-End Orchestration:** Simplifies and automates the orchestration of complex ML pipelines.
*   **Experimentation:** Makes it easy to experiment with different models, techniques, and parameters.
*   **Reusability:** Allows for the reuse of components and pipelines, accelerating development and deployment.
*   **Scalability:** Built on Kubernetes, ensuring scalability for your ML workloads.
*   **Container Runtime Agnostic:** Supports various container runtimes through the Emissary Executor.

## Installation

*   Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, you can deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) as a standalone service.

## Documentation and Resources

*   **Getting Started:** [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Documentation:** [Kubeflow Pipelines SDK Documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API Doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-Powered Documentation:** [DeepWiki](https://deepwiki.com/kubeflow/pipelines)

## Contributing

Contribute to Kubeflow Pipelines.  Read the guidelines in [How to Contribute](./CONTRIBUTING.md) and review the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:**  Every other Wednesday 10-11AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  #kubeflow-pipelines on the Cloud Native Computing Foundation Slack workspace.  Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

*   Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.