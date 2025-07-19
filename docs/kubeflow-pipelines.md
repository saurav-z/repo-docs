# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows on Kubernetes

**(Visit the original repository: [Kubeflow Pipelines](https://github.com/kubeflow/pipelines))**

Kubeflow Pipelines empowers data scientists and engineers to build, deploy, and manage end-to-end machine learning (ML) workflows on Kubernetes, streamlining the ML lifecycle.

**Key Features:**

*   **End-to-End Orchestration:** Simplify the creation and management of complex ML pipelines.
*   **Experimentation:** Easily iterate on ideas, compare techniques, and manage ML trials.
*   **Reusability:** Reuse components and pipelines to accelerate solution development.
*   **Container Runtime Agnostic:** Run Kubeflow Pipelines on Kubernetes clusters with any [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)

**Installation:**

*   Can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy as a standalone service following the instructions in the [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) documentation.

**Documentation:**

*   **Overview:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK:** [Use the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Kubeflow Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-powered documentation:** Check out the AI Powered repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

**Contributing:**

*   Review the [contribution guidelines](./CONTRIBUTING.md).
*   Consult the [developer guide](./developer_guide.md) for building and deploying from source.

**Community:**

*   **Community Meeting:** Occurs every other Wednesday, 10-11AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Join the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace ([Kubeflow Slack Channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)).

**Architecture:**

*   Read more in [Architecture.md](docs/Architecture.md)

**Blog Posts:**

*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

**Acknowledgments:**

*   Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows) by default.