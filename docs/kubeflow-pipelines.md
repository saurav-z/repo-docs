# Kubeflow Pipelines: Build, Deploy, and Manage Machine Learning Workflows

**Orchestrate end-to-end machine learning workflows with ease using Kubeflow Pipelines, a powerful toolkit for streamlining your ML deployments.**  [View the original repository](https://github.com/kubeflow/pipelines)

[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/pipelines)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)

## Key Features of Kubeflow Pipelines

*   **End-to-End Orchestration:** Simplify the management and automation of complex machine learning pipelines.
*   **Easy Experimentation:**  Accelerate your development with a platform to efficiently test, track, and manage multiple trials and ML techniques.
*   **Component and Pipeline Reusability:** Save time and effort by reusing components and pipelines to quickly build end-to-end solutions, eliminating the need to rebuild from scratch.
*   **Kubernetes Native:** Integrates seamlessly with Kubernetes for scalable and portable ML deployments.

## Getting Started with Kubeflow Pipelines

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.  Refer to the following resources for installation and usage:

*   **Installation:** See the [Kubeflow Pipelines installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) for detailed instructions.
*   **Documentation:** Explore the [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/) to get started and learn more.
*   **SDK Documentation:** Discover various ways to [use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/).
*   **API Reference:** Consult the [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specifications.
*   **Python SDK Reference:** Review the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.

## Deep Wiki

Explore AI-generated documentation with [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning:  Please note, this is AI-generated content and may contain inaccuracies.

## Contributing

Contribute to the Kubeflow Pipelines project:

*   **Contribution Guidelines:** Review the [CONTRIBUTING.md](./CONTRIBUTING.md) file for detailed contribution guidelines.
*   **Developer Guide:** Consult the [developer_guide.md](./developer_guide.md) for building and deploying Kubeflow Pipelines from source code.

## Community

Connect with the Kubeflow Pipelines community:

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace.  More details are available at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Learn more about the architecture in [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) under the hood. The Argo community's support is greatly appreciated.