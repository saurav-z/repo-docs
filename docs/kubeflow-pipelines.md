# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Kubeflow Pipelines empowers data scientists and machine learning engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes.** ([Original Repository](https://github.com/kubeflow/pipelines))

[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/pipelines)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)

## Key Features

*   **End-to-End Orchestration:** Simplify the creation and management of complex ML pipelines.
*   **Experimentation Made Easy:**  Easily try out new ideas and techniques, and manage experiments efficiently.
*   **Component and Pipeline Reusability:** Build upon existing components and pipelines, reducing development time and promoting code reuse.
*   **Scalability:** Leverage Kubernetes for scalable and reliable ML workflow execution.
*   **Container Runtime Agnostic:** Supports a variety of container runtimes.

## Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.
See the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) for detailed instructions.

## Documentation and Resources

*   **Kubeflow Pipelines Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **Kubeflow Pipelines SDK:** [Use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **Kubeflow Pipelines API:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community and Contributing

*   **Contributing:** Review the [How to Contribute](./CONTRIBUTING.md) guidelines before contributing.
*   **Developer Guide:** Learn how to build and deploy Kubeflow Pipelines from source code with the [developer guide](./developer_guide.md).
*   **Community Meeting:**  Every other Wednesday, 10-11AM (PST) - [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Meeting notes](http://bit.ly/kfp-meeting-notes).
*   **Slack:** Join the Kubeflow Pipelines channel (#kubeflow-pipelines) on the Cloud Native Computing Foundation Slack workspace ([Kubeflow Slack Channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels))

## Architecture

*   For details on the architecture, see [Architecture.md](docs/Architecture.md).

## Blog Posts and Articles

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.