# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**Kubeflow Pipelines empowers you to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, simplifying the ML lifecycle.** ([See the original repository](https://github.com/kubeflow/pipelines))

[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/pipelines?branch=master)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Streamlines the entire ML pipeline lifecycle, from data ingestion to model deployment.
*   **Easy Experimentation:** Facilitates rapid experimentation and tracking of different ML techniques and configurations.
*   **Component & Pipeline Reusability:** Enables the reuse of pre-built components and pipelines, accelerating development.
*   **Scalability and Portability:** Leverages Kubernetes to provide scalable and portable ML workflows.
*   **Container Runtime Agnostic:** Runs on Kubernetes clusters with any container runtime.

## Getting Started

### Installation

*   Install Kubeflow Pipelines as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy it as a standalone service using the [installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

### Documentation and Resources

*   **Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Documentation:** [Use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Community and Contribution

*   **Contributing:**  Review the [contribution guidelines](./CONTRIBUTING.md)
*   **Developer Guide:** Learn to build and deploy Kubeflow Pipelines from source code via the [developer guide](./developer_guide.md)
*   **Community Meetings:** Attend bi-weekly meetings - details and links are included in the original README.
*   **Slack:** Join the `#kubeflow-pipelines` channel on the Cloud Native Computing Foundation Slack workspace ([more details](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)).

## Additional Resources

*   **Deep Wiki:** Explore AI-powered documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).  (*Note: Information accuracy may vary.*)
*   **Architecture:**  Understand the architecture of Kubeflow Pipelines: [Architecture.md](docs/Architecture.md)
*   **Blog Posts:** Explore blog posts for real-world use cases and tutorials (links in original README).

## Acknowledgments

Kubeflow Pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestration, and we are grateful for the Argo community's support.