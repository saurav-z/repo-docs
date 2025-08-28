# Kubeflow Pipelines: Automate and Orchestrate Your Machine Learning Workflows

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes.**  Explore the original repo at [https://github.com/kubeflow/pipelines](https://github.com/kubeflow/pipelines).

<!-- Badges - can be included if desired, but removed for clarity -->
<!--
[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/kubeflow/pipelines?branch=master)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)
-->

## Key Features

*   **End-to-End Orchestration:** Simplify and automate the entire machine learning pipeline lifecycle, from data ingestion to model deployment.
*   **Experimentation and Iteration:** Easily experiment with different models, techniques, and parameters, and manage your experiments effectively.
*   **Component and Pipeline Reusability:** Build and reuse modular components and complete pipelines to accelerate development and avoid redundant work.
*   **Scalability and Portability:** Leverage the power of Kubernetes to scale your machine learning workloads and ensure portability across different environments.

## Installation

Kubeflow Pipelines can be installed in a couple of ways:

*   As part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   As a standalone service, following the instructions in the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

### Dependencies Compatibility Matrix

| Dependency     | Versions    |
| -------------- | ----------  |
| Argo Workflows | v3.5, v3.6  |
| MySQL          | v8          |

## Documentation and Resources

*   **Overview:** Get started with the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/) to understand the core concepts.
*   **SDK Documentation:** Learn how to use the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/) to build your pipelines.
*   **API Reference:** Explore the [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for detailed API specifications.
*   **Python SDK Reference:** Find in-depth information in the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) while using the Python SDK.

## Deep Wiki

Check out our AI-powered repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning: Please note, this is AI-generated and may not have completely accurate information.

## Contributing

Read the [contributing guidelines](./CONTRIBUTING.md) to start contributing to Kubeflow Pipelines.  For information on building and deploying from source, see the [developer guide](./developer_guide.md).

## Community

### Community Meetings

Join the Kubeflow Pipelines Community Meeting every other Wednesday from 10-11 AM (PST).

*   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
*   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
*   [Meeting notes](http://bit.ly/kfp-meeting-notes)

### Slack

Join the Kubeflow Pipelines channel (#kubeflow-pipelines) on the Cloud Native Computing Foundation Slack workspace.  Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels).

## Architecture

For details about the KFP Architecture, refer to the [Architecture.md](docs/Architecture.md) file.

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) under the hood to orchestrate Kubernetes resources. We are grateful to the Argo community for their support.