# Kubeflow Pipelines: Simplify and Automate Your Machine Learning Workflows

**Kubeflow Pipelines empowers data scientists and ML engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes.**  ([Original Repository](https://github.com/kubeflow/pipelines))

## Key Features

*   **End-to-End Orchestration:**  Orchestrates complex ML pipelines, streamlining the entire ML lifecycle.
*   **Experimentation:**  Facilitates easy experimentation and management of various ML trials and techniques.
*   **Reusability:**  Enables the reuse of components and pipelines, accelerating development and reducing redundant work.
*   **Scalability:** Built on Kubernetes for scalability to handle large datasets and complex model training.
*   **Container Compatibility:** Uses the Emissary Executor by default, enabling Kubeflow Pipelines to run on any Kubernetes cluster, regardless of the container runtime.

## What are Kubeflow Pipelines?

Kubeflow Pipelines are reusable ML workflows constructed using the Kubeflow Pipelines SDK. They are a core component of [Kubeflow](https://www.kubeflow.org/), a toolkit designed to simplify and accelerate the deployment of ML workflows on Kubernetes. This includes model training, hyperparameter tuning, model serving, and more.

## Getting Started

*   **Installation:** Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or deployed as a standalone service.
*   **Documentation:**
    *   [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
    *   [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
    *   [Pipelines API Documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
    *   [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community & Contribution

*   **Contributing:** Review the guidelines in [How to Contribute](./CONTRIBUTING.md).
*   **Developer Guide:** Learn how to build and deploy Kubeflow Pipelines from source code [developer guide](./developer_guide.md).
*   **Community Meeting:**  Bi-weekly meetings every other Wednesday, 10-11 AM PST ([Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVaZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Meeting notes](http://bit.ly/kfp-meeting-notes))
*   **Slack:**  Join the conversation in the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. ([More info](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Architecture

*   For detailed information on the architecture, please refer to [Architecture.md](docs/Architecture.md)

## Additional Resources

*   **Blog Posts:** Access a collection of blog posts offering insights into the use of Kubeflow Pipelines.

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for underlying Kubernetes resource orchestration. We are grateful to the Argo community for their support.