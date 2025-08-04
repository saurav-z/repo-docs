# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows

**Kubeflow Pipelines empowers you to create reproducible and scalable machine learning workflows on Kubernetes.**  For more information, see the original repository at [https://github.com/kubeflow/pipelines](https://github.com/kubeflow/pipelines).

## Key Features

*   **End-to-end Orchestration:** Simplify the management of your complete machine learning pipelines.
*   **Easy Experimentation:**  Quickly test and iterate on various ML techniques and manage your experiments effectively.
*   **Component and Pipeline Reusability:**  Build solutions faster by reusing components and pipelines, avoiding repetitive builds.
*   **Container Runtime Agnostic:**  Kubeflow Pipelines supports various [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/).

## Overview

Kubeflow Pipelines is an essential component of the [Kubeflow](https://www.kubeflow.org/) toolkit, designed to streamline machine learning (ML) deployments on Kubernetes. It offers a platform for building, deploying, and managing end-to-end ML workflows, promoting reusability, and ease of experimentation.

## Installation

*   Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.
*   The Emissary Executor is the default executor from Kubeflow Pipelines 1.8, which is Container runtime agnostic.

## Documentation

*   **Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:** [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-Powered Documentation:** [DeepWiki](https://deepwiki.com/kubeflow/pipelines)

## Contributing

Contribute to Kubeflow Pipelines by following the guidelines in [How to Contribute](./CONTRIBUTING.md) and learn how to build and deploy from source using the [developer guide](./developer_guide.md).

## Community

*   **Community Meeting:** Every other Wednesday, 10-11 AM (PST)
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:**  Find us in the `#kubeflow-pipelines` channel on the Cloud Native Computing Foundation Slack workspace.  Details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

*   Details about the KFP Architecture can be found at [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestration.