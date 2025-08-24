# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning (ML) workflows with ease.**

[View the original repository on GitHub](https://github.com/kubeflow/pipelines)

## Key Features

*   **End-to-End Orchestration:** Simplify the complex process of orchestrating ML pipelines.
*   **Experimentation:** Easily try out different ideas, techniques, and manage multiple trials and experiments.
*   **Reusability:** Promote code reuse and accelerate development by leveraging reusable components and pipelines.
*   **Kubernetes Native:** Built to run seamlessly on Kubernetes, leveraging its scalability and portability.
*   **Flexible Execution:** Supports various Container runtimes with the Emissary Executor, including Docker.

## What is Kubeflow Pipelines?

Kubeflow Pipelines is a core component of [Kubeflow](https://www.kubeflow.org/), a toolkit designed to simplify the deployment of ML workflows on Kubernetes. Kubeflow Pipelines allows you to create and manage reusable ML workflows, enabling faster experimentation, streamlined deployments, and improved collaboration.

## Getting Started

### Installation

*   Integrate Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or deploy it as a standalone service using the [installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

### Documentation & Resources

*   **Overview:** Get started with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK:** Learn how to [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API:** Explore the [Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/).
*   **SDK Reference:** Consult the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.

## Community

*   **Community Meetings:** Join the Kubeflow Pipelines Community Meeting every other Wednesday at 10-11AM PST.
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Connect with the community on the [Cloud Native Computing Foundation Slack workspace](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) in the #kubeflow-pipelines channel.

## Contributing

Contribute to the project following the guidelines in [How to Contribute](./CONTRIBUTING.md). For building and deploying from source, see the [developer guide](./developer_guide.md).

## Further Information

*   **Architecture:** Dig into the KFP Architecture at [Architecture.md](docs/Architecture.md).
*   **Blog Posts:** Explore relevant blog posts for practical insights.

## Acknowledgements

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) for Kubernetes resource orchestration.