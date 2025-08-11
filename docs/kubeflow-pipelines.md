# Kubeflow Pipelines: Build, Deploy, and Manage End-to-End Machine Learning Workflows

**Kubeflow Pipelines** empowers you to orchestrate and automate your entire machine learning lifecycle, from data ingestion to model deployment, all within your Kubernetes environment.  Explore the full capabilities on [GitHub](https://github.com/kubeflow/pipelines).

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Simplify the complex process of building and managing your machine learning pipelines.
*   **Simplified Experimentation:** Easily test and iterate on various ideas and techniques with robust experiment management.
*   **Component and Pipeline Reusability:** Accelerate development by leveraging reusable components and pipelines, reducing the need to rebuild from scratch.
*   **Kubernetes Native:** Seamlessly integrates with Kubernetes for scalability, portability, and efficient resource utilization.
*   **Flexible Execution:** Supports the Emissary Executor, allowing you to run Kubeflow Pipelines on Kubernetes clusters with diverse container runtimes.

## Installation

Kubeflow Pipelines can be installed either as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.  Detailed installation guides are available in the [Kubeflow Pipelines documentation](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation and Resources

*   **Kubeflow Pipelines Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Documentation:** [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Documentation:** [Kubeflow Pipelines API](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-Powered Repo Documentation:** [DeepWiki](https://deepwiki.com/kubeflow/pipelines)

## Contributing

Learn how to contribute to Kubeflow Pipelines by reading the guidelines in [How to Contribute](./CONTRIBUTING.md). To learn how to build and deploy Kubeflow Pipelines from source code, read the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST)
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  #kubeflow-pipelines on the Cloud Native Computing Foundation Slack workspace.  Find details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

For detailed information about the KFP Architecture, see [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.