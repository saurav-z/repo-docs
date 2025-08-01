# Kubeflow Pipelines: Automate and Scale Your Machine Learning Workflows

Kubeflow Pipelines empowers data scientists and engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes.  [Learn more on the original repository](https://github.com/kubeflow/pipelines).

[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/pipelines?branch=master)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Simplify the creation and management of complex ML pipelines.
*   **Experimentation:** Easily test and iterate on different ML techniques.
*   **Component Reusability:** Build pipelines from reusable components, saving time and effort.
*   **Scalability:** Leverage Kubernetes to scale your ML workflows.
*   **Portability:** Run your pipelines on any Kubernetes cluster.

## Getting Started with Kubeflow Pipelines

### Installation

*   Install as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.
*   Kubeflow Pipelines uses the [Emissary Executor](https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/choose-executor/#emissary-executor) by default, compatible with various container runtimes.

### Documentation

*   **Overview:**  [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK:**  [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API:** [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community & Resources

*   **DeepWiki:** Explore AI-powered documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).
*   **Contributing:**  Review the guidelines in [How to Contribute](./CONTRIBUTING.md).
*   **Developer Guide:** Learn to build and deploy from source [developer guide](./developer_guide.md).
*   **Community Meeting:** Bi-weekly community meeting (every other Wednesday, 10-11 AM PST) - [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL), [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x), [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace: [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

*   Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts & Articles

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
  * [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
  * [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
  * [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.