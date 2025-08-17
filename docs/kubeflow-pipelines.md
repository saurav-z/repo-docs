# Kubeflow Pipelines: Build, Deploy, and Manage Scalable Machine Learning Workflows

**Kubeflow Pipelines empowers data scientists and ML engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, streamlining the ML lifecycle for increased efficiency and reproducibility.** ([Original Repo](https://github.com/kubeflow/pipelines))

## Key Features of Kubeflow Pipelines:

*   **End-to-End Orchestration:** Simplify the creation and management of complex ML pipelines, from data ingestion to model deployment.
*   **Easy Experimentation:** Facilitate rapid prototyping and experimentation with flexible pipeline design and versioning.
*   **Component and Pipeline Reusability:**  Accelerate development by enabling the reuse of pipeline components and complete pipelines.
*   **Kubernetes Native:** Leverage the power and scalability of Kubernetes for ML workload orchestration.
*   **Integration with Kubeflow:** Seamlessly integrates with other Kubeflow components for a comprehensive ML platform.

## Getting Started

### Installation
Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service ([Kubeflow Pipelines Installation Guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)).

### Documentation
*   **Kubeflow Pipelines Overview:** [https://www.kubeflow.org/docs/components/pipelines/overview/](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **Kubeflow Pipelines SDK:** [https://kubeflow-pipelines.readthedocs.io/en/stable/](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **Pipelines API Doc:** [https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference Docs:** [https://kubeflow-pipelines.readthedocs.io/en/stable/](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community and Contribution

*   **Contributing:**  Read the guidelines in [How to Contribute](./CONTRIBUTING.md).
*   **Developer Guide:** Learn how to build and deploy Kubeflow Pipelines from source code: [developer_guide.md](./developer_guide.md).
*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST) - [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the `#kubeflow-pipelines` channel on the Cloud Native Computing Foundation Slack workspace: [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Detailed information about the Kubeflow Pipelines architecture can be found in [Architecture.md](docs/Architecture.md).

## Additional Resources

*   **Blog Posts:** Explore a range of blog posts covering different aspects of Kubeflow Pipelines from the official blog and contributors:
    *   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/)
    *   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines)
    *   How to create and deploy a Kubeflow Machine Learning Pipeline (Part 1, 2, and 3)

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestrating Kubernetes resources, and we are grateful for the Argo community's support.