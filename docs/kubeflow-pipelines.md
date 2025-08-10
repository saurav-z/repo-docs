# Kubeflow Pipelines: Orchestrate, Experiment, and Scale Your Machine Learning Workflows

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning workflows, simplifying and accelerating your ML lifecycle.** You can find the original repository [here](https://github.com/kubeflow/pipelines).

## Key Features and Benefits

*   **End-to-End Orchestration:** Streamline your ML pipelines by orchestrating complex workflows from data ingestion to model deployment.
*   **Simplified Experimentation:** Easily test and iterate on various ML ideas and techniques, facilitating experimentation and rapid prototyping.
*   **Reusable Components:** Leverage pre-built components and pipelines to accelerate development and reduce code duplication.
*   **Scalability:** Run your ML pipelines at scale using Kubernetes, ensuring efficient resource utilization and handling of large datasets.
*   **Portability:** Deploy your ML workflows across different environments, promoting flexibility and adaptability.

## Installation

*   Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) as a standalone service.
*   Kubeflow Pipelines uses the [Emissary Executor](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/choose-executor/#emissary-executor) by default, making it compatible with any [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/) on your Kubernetes cluster.

## Documentation

*   **Kubeflow Pipelines Overview:** Get started with your first pipeline and read further information in the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Explore the various ways you can [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Consult the Kubeflow [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specification.
*   **Python SDK Reference:** Use the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.

## Community and Resources

*   **Deep Wiki:** Explore AI-powered documentation with [DeepWiki](https://deepwiki.com/kubeflow/pipelines).
*   **Contributing:** Read the guidelines in [How to Contribute](./CONTRIBUTING.md) and learn how to build and deploy Kubeflow Pipelines from source code in the [developer guide](./developer_guide.md).
*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday, 10-11 AM (PST) [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x), [Meeting notes](http://bit.ly/kfp-meeting-notes).
*   **Slack:** Connect with the community in the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace ([https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)).

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

Kubeflow pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources, and the Argo community's support is greatly appreciated.