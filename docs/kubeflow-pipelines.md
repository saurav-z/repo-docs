# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Simplify and accelerate your machine learning lifecycle with Kubeflow Pipelines, the powerful platform for building, deploying, and managing end-to-end ML workflows on Kubernetes.**

[View the original repository on GitHub](https://github.com/kubeflow/pipelines)

## Key Features

*   **End-to-End Orchestration:** Orchestrate complex ML workflows, from data ingestion to model deployment, with ease.
*   **Simplified Experimentation:** Easily experiment with different ML techniques, track results, and manage your ML trials.
*   **Reusable Components:** Build reusable pipeline components to accelerate development and reduce code duplication.
*   **Scalability:** Leverage the power of Kubernetes to scale your ML pipelines to meet your growing needs.

## Overview

Kubeflow Pipelines is a core component of Kubeflow, a machine learning (ML) toolkit designed to simplify and streamline deployments of ML workflows on Kubernetes. It enables you to create reusable, end-to-end ML workflows using the Kubeflow Pipelines SDK. This allows data scientists and ML engineers to automate and manage their ML pipelines, improving efficiency and reproducibility.

## Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.

*   Kubeflow Pipelines now defaults to using the [Emissary Executor](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/choose-executor/#emissary-executor), which is container runtime agnostic.

## Documentation

*   [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   [Pipelines API Reference](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community

*   **Community Meetings:** Every other Wednesday, 10-11 AM (PST)
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Find us in the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace ([https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Contributing

Contribute to Kubeflow Pipelines by reviewing the guidelines in [How to Contribute](./CONTRIBUTING.md). Learn how to build and deploy Kubeflow Pipelines from source code using the [developer guide](./developer_guide.md).

## Architecture

Learn about the KFP architecture: [Architecture.md](docs/Architecture.md)

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources.