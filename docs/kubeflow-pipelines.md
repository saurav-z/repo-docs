# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Orchestrate your machine learning pipelines with ease using Kubeflow Pipelines, the open-source solution designed for scalability and reproducibility.**  [View the original repository on GitHub](https://github.com/kubeflow/pipelines).

![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)
![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)
![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)
![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)
![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)
![Ask DeepWiki](https://deepwiki.com/badge.svg)

## Key Features of Kubeflow Pipelines

Kubeflow Pipelines offers a comprehensive platform for managing your ML workflows, providing:

*   **End-to-End Orchestration:** Simplifies the management of complex ML pipelines from data ingestion to model deployment.
*   **Experiment Tracking & Management:** Facilitates easy experimentation with built-in support for versioning, comparison, and reproducibility of your trials.
*   **Component & Pipeline Reusability:** Enables the creation of reusable components and pipelines, accelerating development and minimizing redundant work.
*   **Scalability:** Leverage the power of Kubernetes for scalable and efficient ML workflow execution.

## Getting Started with Kubeflow Pipelines

### Installation

You can install Kubeflow Pipelines in two main ways:

*   **As part of the Kubeflow Platform:**  Follow the installation guide available in the [Kubeflow documentation](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   **As a Standalone Service:** Deploy Kubeflow Pipelines independently using the instructions provided in the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

**Important Note:** Kubeflow Pipelines utilizes the Emissary Executor by default since Kubeflow Pipelines 1.8. This container runtime agnostic executor ensures compatibility across various Kubernetes container runtimes.

### Dependencies Compatibility Matrix

| Dependency     | Versions    |
| -------------- | ----------  |
| Argo Workflows | v3.5, v3.6  |
| MySQL          | v8          |

## Resources & Documentation

*   **Overview:**  Get a high-level understanding of Kubeflow Pipelines in the [official documentation](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Learn how to use the SDK for building and interacting with pipelines by visiting the [Kubeflow Pipelines SDK documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Consult the [Kubeflow Pipelines API documentation](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for detailed API specifications.
*   **Python SDK Reference:** Use the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) to streamline the pipeline creation process.
*   **AI-Powered Documentation:** Discover AI-generated documentation using [DeepWiki](https://deepwiki.com/kubeflow/pipelines) for supplementary information.
> :warning: Please note, this is AI generated and may not have completely accurate information.

## Contribute to Kubeflow Pipelines

Contribute to the evolution of Kubeflow Pipelines by reviewing the [contribution guidelines](./CONTRIBUTING.md).  You can also learn how to build and deploy Kubeflow Pipelines from the source code by exploring the [developer guide](./developer_guide.md).

## Connect with the Kubeflow Pipelines Community

*   **Community Meeting:**  Join the Kubeflow Pipelines Community Meeting every other Wednesday from 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Engage with the community on the #kubeflow-pipelines channel within the Cloud Native Computing Foundation Slack workspace ([https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)).

## Architecture

For in-depth details of the KFP architecture, please review the [Architecture.md](docs/Architecture.md) document.

## Blog Posts & Resources

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines is built on top of [Argo Workflows](https://github.com/argoproj/argo-workflows), a powerful workflow engine.  We are immensely grateful to the Argo community for their ongoing support.