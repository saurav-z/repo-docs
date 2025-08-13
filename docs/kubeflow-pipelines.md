# Kubeflow Pipelines: Build, Deploy, and Manage Scalable ML Workflows

[![Coverage Status](https://coveralls.io/repos/github/kubeflow/pipelines/badge.svg?branch=master)](https://coveralls.io/github/kubeflow/pipelines?branch=master)
[![SDK Documentation Status](https://readthedocs.org/projects/kubeflow-pipelines/badge/?version=latest)](https://kubeflow-pipelines.readthedocs.io/en/stable/?badge=latest)
[![SDK Package version](https://img.shields.io/pypi/v/kfp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kfp)
[![SDK Supported Python versions](https://img.shields.io/pypi/pyversions/kfp.svg?color=%2334D058)](https://pypi.org/project/kfp)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9938/badge)](https://www.bestpractices.dev/projects/9938)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/pipelines)

**Kubeflow Pipelines simplifies and accelerates the machine learning lifecycle by providing a platform for building, deploying, and managing end-to-end ML workflows.**  [Explore the original repository](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-End Orchestration:** Simplify the orchestration of complex machine learning pipelines.
*   **Experimentation Made Easy:**  Facilitate the rapid testing of ideas and techniques, and manage various trials and experiments effectively.
*   **Component and Pipeline Reusability:**  Accelerate development with reusable components and pipelines, eliminating the need for repetitive builds.
*   **Container Runtime Agnostic:** The Emissary Executor is used by default, allowing Kubeflow Pipelines to run on Kubernetes clusters with any container runtime.

## Overview

Kubeflow Pipelines is a powerful machine learning toolkit built on top of [Kubeflow](https://www.kubeflow.org/), designed to streamline the deployment of ML workflows on Kubernetes.  It empowers data scientists and ML engineers to build portable, scalable, and reproducible ML pipelines.

## Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service. See the [installation guide](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) for detailed instructions.

## Documentation

Comprehensive documentation is available to help you get started and dive deeper:

*   **Getting Started:**  [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:**  [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:**  [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK Reference:**  [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)

## Deep Wiki (AI-Powered Documentation)

Explore AI-generated documentation for Kubeflow Pipelines on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

> :warning:  Please note that this is AI-generated and may not have completely accurate information.

## Contributing

We welcome contributions!  Please review the guidelines in [How to Contribute](./CONTRIBUTING.md) and the [developer guide](./developer_guide.md) to learn how to build and deploy Kubeflow Pipelines from source code.

## Community

Connect with the Kubeflow Pipelines community:

*   **Community Meetings:**  Every other Wednesday, 10-11 AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting Notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  Join the `#kubeflow-pipelines` channel on the Cloud Native Computing Foundation Slack workspace.  Find details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Learn more about the Kubeflow Pipelines architecture in [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) by default. We are grateful for the support of the Argo community.