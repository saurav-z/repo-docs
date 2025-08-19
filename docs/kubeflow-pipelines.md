# Kubeflow Pipelines: Build, Deploy, and Manage ML Workflows at Scale

**Kubeflow Pipelines empowers data scientists and machine learning engineers to build, deploy, and manage end-to-end machine learning workflows with ease.** [Learn more on GitHub](https://github.com/kubeflow/pipelines).

## Key Features:

*   **End-to-End Orchestration:** Simplify the process of orchestrating complex machine learning pipelines.
*   **Experiment Tracking:** Easily experiment with different ideas and techniques, managing your trials and experiments effectively.
*   **Component and Pipeline Reusability:** Accelerate development by reusing components and pipelines, avoiding redundant builds.
*   **Scalability:** Designed to handle the demands of large-scale machine learning deployments on Kubernetes.
*   **Container Runtime Agnostic**: Runs on Kubernetes clusters with any [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)

## Installation

*   Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform). Alternatively you can deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) as a standalone service.

## Documentation and Resources

*   **Kubeflow Pipelines Overview:** Get started and learn the fundamentals at [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/).
*   **SDK Documentation:** Explore the [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/) for building pipelines.
*   **API Reference:** Consult the [Pipelines API](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/) for detailed API specifications.
*   **Python SDK Reference:** Refer to the [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/) for writing pipelines using the Python SDK.

## Community

*   **Community Meeting:** Join the Kubeflow Pipelines Community Meeting every other Wednesday, 10-11 AM PST.
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Connect with the community on the Kubeflow Pipelines Slack channel (#kubeflow-pipelines) on the Cloud Native Computing Foundation Slack workspace. More details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Additional Resources

*   **Architecture:** Dive into the KFP architecture at [docs/Architecture.md](docs/Architecture.md)
*   **Blog Posts:** Explore insightful blog posts on Kubeflow Pipelines:
    *   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/)
    *   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines)
    *   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
        *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
        *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
        *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Contributing

Contribute to Kubeflow Pipelines by reading the guidelines in [How to Contribute](./CONTRIBUTING.md). Learn how to build and deploy from source code via the [developer guide](./developer_guide.md).

## Acknowledgments

Kubeflow pipelines uses [Argo Workflows](https://github.com/argoproj/argo-workflows) by default under the hood to orchestrate Kubernetes resources. The Argo community has been very supportive and we are very grateful.