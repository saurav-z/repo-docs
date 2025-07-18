# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows

**Kubeflow Pipelines simplifies and accelerates machine learning workflows, empowering you to build, deploy, and manage end-to-end ML pipelines at scale.**  For the original source code, see the [Kubeflow Pipelines repository](https://github.com/kubeflow/pipelines).

## Key Features:

*   **End-to-End Orchestration:** Streamline and simplify the orchestration of your entire machine learning lifecycle.
*   **Simplified Experimentation:** Easily experiment with different ideas and techniques, managing trials and experiments effectively.
*   **Component and Pipeline Reusability:** Build upon existing components and pipelines for rapid development and deployment, reducing the need for repetitive building.
*   **Container Runtime Agnostic:** Compatible with various container runtimes, offering flexibility in your Kubernetes environment.

## Installation

*   Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) as a standalone service.

## Documentation

*   **Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:** [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. More details: [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Additional Resources

*   **Architecture:** [Architecture.md](docs/Architecture.md)
*   **Blog Posts:**
    *   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
    *   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
        *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
        *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
        *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)
*   **Deep Wiki:** Check out our AI Powered repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

    > :warning: Please note, this is AI generated and may not have completely accurate information.

## Contributing

Review the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines before contributing. To build and deploy from source, see the [developer guide](./developer_guide.md).

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for Kubernetes resource orchestration. The Argo community's support is greatly appreciated.