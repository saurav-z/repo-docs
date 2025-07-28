# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows

**Kubeflow Pipelines empowers data scientists and ML engineers to build, deploy, and manage end-to-end machine learning workflows on Kubernetes, enabling faster experimentation and improved model deployment.**

[Visit the original Kubeflow Pipelines repository](https://github.com/kubeflow/pipelines)

Kubeflow Pipelines is a core component of the Kubeflow ML toolkit, designed to simplify and accelerate the ML lifecycle. It allows you to build reproducible and scalable ML pipelines, streamlining the development, deployment, and management of your ML models.

**Key Features:**

*   **End-to-End Orchestration:**  Orchestrate complex ML pipelines from data preparation to model deployment.
*   **Simplified Experimentation:** Easily experiment with different techniques and model variations.
*   **Reusable Components:**  Build modular components and pipelines for rapid development and deployment.
*   **Container Runtime Agnostic**: Run Kubeflow Pipelines on Kubernetes clusters with any container runtimes.

## Getting Started

### Installation

Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform) or as a standalone service.

## Documentation and Resources

*   **Kubeflow Pipelines Overview:**  Learn more about Kubeflow Pipelines in the official [overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Explore the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/) for building pipelines.
*   **API Documentation:** Reference the [Kubeflow Pipelines API](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for API specifications.
*   **Python SDK Reference:** Consult the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) when writing pipelines using the Python SDK.
*   **Deep Wiki:** Explore the AI-powered documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).

## Contributing

Interested in contributing to Kubeflow Pipelines?  Review the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.
For information on building and deploying Kubeflow Pipelines from source code, see the [developer guide](./developer_guide.md).

## Community

### Community Meetings

Join the Kubeflow Pipelines Community Meeting every other Wednesday from 10-11 AM (PST).

*   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
*   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
*   [Meeting notes](http://bit.ly/kfp-meeting-notes)

### Slack

Connect with the community on the Cloud Native Computing Foundation Slack workspace in the #kubeflow-pipelines channel.  Find more details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

Detailed information about the Kubeflow Pipelines architecture can be found in [Architecture.md](docs/Architecture.md).

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines utilizes [Argo Workflows](https://github.com/argoproj/argo-workflows), and we are grateful to the Argo community for their support.