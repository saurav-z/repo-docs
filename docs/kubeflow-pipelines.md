# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Kubeflow Pipelines** empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, streamlining your ML lifecycle.  [Learn more at the Kubeflow Pipelines repository](https://github.com/kubeflow/pipelines).

[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/kubeflow/pipelines?branch=master)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)

## Key Features of Kubeflow Pipelines:

*   **End-to-End Orchestration:** Simplify the management of complex machine learning pipelines, from data ingestion to model deployment.
*   **Easy Experimentation:** Rapidly iterate on your ML ideas and techniques with built-in support for managing experiments and trials.
*   **Component & Pipeline Reusability:** Accelerate development by reusing pre-built components and pipelines, reducing the need for repetitive coding.
*   **Kubernetes Native:** Built to run on Kubernetes, providing scalability and portability for your ML workloads.
*   **Container Runtime Agnostic:** Works with various container runtimes on Kubernetes.

## Getting Started

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.  Refer to the [installation documentation](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) for detailed instructions.

## Documentation & Resources

*   **Overview:** [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Documentation:** [Kubeflow Pipelines SDK Documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK Reference Docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **DeepWiki:** Explore AI-powered documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning: Please note, this is AI generated and may not have completely accurate information.

## Contribute

Contribute to the project by following the guidelines in [How to Contribute](./CONTRIBUTING.md). To build and deploy Kubeflow Pipelines from source code, refer to the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday at 10-11 AM (PST).  [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL) and [Meeting notes](http://bit.ly/kfp-meeting-notes).
*   **Slack:** Connect with the community on the `#kubeflow-pipelines` channel in the Cloud Native Computing Foundation Slack workspace.  Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

*   **Architecture:** Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts & Tutorials

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/)
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines)
*   How to create and deploy a Kubeflow Machine Learning Pipeline:
    *   [Part 1](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines relies on [Argo Workflows](https://github.com/argoproj/argo-workflows).  We are grateful for the Argo community's support.