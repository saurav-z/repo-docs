# Kubeflow Pipelines: Automate and Scale Your Machine Learning Workflows

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, streamlining your ML lifecycle.** [Explore the original repository](https://github.com/kubeflow/pipelines).

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Simplify and automate the orchestration of complex ML pipelines.
*   **Easy Experimentation:** Rapidly test new ideas, techniques, and manage various experiments.
*   **Component and Pipeline Reusability:**  Build on existing components and pipelines, minimizing redundant development.
*   **Scalability:** Leverage the power of Kubernetes to scale your ML workflows efficiently.
*   **Container Runtime Agnostic:**  Runs on Kubernetes clusters with any container runtime.

## Installation

*   Install as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Deploy as a standalone service using the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

## Documentation & Resources

*   **Overview:** Get started with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK:** Learn to [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Explore the [Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/).
*   **Python SDK:** Consult the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.
*   **Deep Wiki:**  Explore AI-generated documentation via [DeepWiki](https://deepwiki.com/kubeflow/pipelines). (Note: Information may not be completely accurate).

## Contributing

*   Read the [How to Contribute](./CONTRIBUTING.md) guidelines.
*   Learn how to build and deploy from source code with the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace ([https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)).

## Architecture

*   Find architectural details in [Architecture.md](docs/Architecture.md).

## Blog Posts

*   A list of blog posts about Kubeflow Pipelines is available.

## Acknowledgments

*   Kubeflow Pipelines relies on [Argo Workflows](https://github.com/argoproj/argo-workflows) for Kubernetes resource orchestration.