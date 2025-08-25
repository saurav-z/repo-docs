# Kubeflow Pipelines: Orchestrate, Experiment, and Scale Your Machine Learning Workflows

**Kubeflow Pipelines** empowers data scientists and ML engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes. ([See the original repo](https://github.com/kubeflow/pipelines))

Kubeflow Pipelines provides a platform for orchestrating your entire ML lifecycle, from data preparation to model deployment and monitoring.

## Key Features

*   **End-to-End Orchestration:** Simplify and automate the complex process of building and running ML pipelines.
*   **Experimentation:** Easily track, manage, and iterate on your ML experiments with version control and reusability.
*   **Reusability:** Build modular pipeline components that can be easily reused across different projects, saving time and effort.
*   **Scalability:** Leverage the power of Kubernetes to scale your ML pipelines to meet your growing needs.

## Installation

*   **Kubeflow Platform:** Install Kubeflow Pipelines as part of the comprehensive Kubeflow platform.
*   **Standalone Installation:** Deploy Kubeflow Pipelines as a standalone service.
*   **Container Runtime Agnostic:** Kubeflow Pipelines uses the Emissary Executor by default, enabling you to run pipelines on Kubernetes clusters with any supported container runtime.

## Documentation

*   [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   [Kubeflow Pipelines API](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   [Python SDK Reference](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Deep Wiki

Explore AI-powered documentation of Kubeflow Pipelines on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning: Please note, this is AI generated and may not have completely accurate information.

## Contributing

Contribute to Kubeflow Pipelines by reading the guidelines in [How to Contribute](./CONTRIBUTING.md). Learn how to build and deploy Kubeflow Pipelines from source code in the [developer guide](./developer_guide.md).

## Community

### Community Meeting

Join the Kubeflow Pipelines Community Meeting every other Wednesday, 10-11 AM PST.

*   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
*   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
*   [Meeting notes](http://bit.ly/kfp-meeting-notes)

### Slack

Connect with the Kubeflow Pipelines community on the Cloud Native Computing Foundation Slack workspace in the #kubeflow-pipelines channel. Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

For architectural details, see [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default. The Argo community is appreciated.