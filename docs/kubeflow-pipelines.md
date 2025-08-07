# Kubeflow Pipelines: Build, Deploy, and Manage Machine Learning Workflows

**Effortlessly orchestrate and automate your machine learning pipelines with Kubeflow Pipelines, streamlining your ML lifecycle.**  ([View the original repository](https://github.com/kubeflow/pipelines))

Kubeflow Pipelines, built upon the Kubeflow ML toolkit, empowers data scientists and ML engineers to create reproducible and scalable end-to-end machine learning workflows on Kubernetes.

## Key Features of Kubeflow Pipelines:

*   **End-to-End Orchestration:** Simplify the complex process of managing and coordinating your entire ML pipeline, from data ingestion to model deployment.
*   **Experiment Tracking & Management:** Easily try out various ideas and techniques, keeping track of your experiments and trials for optimal model development.
*   **Component & Pipeline Reusability:** Quickly build and reuse components and pipelines to assemble complete ML solutions without starting from scratch.
*   **Container Runtime Agnostic:** Seamlessly run Kubeflow Pipelines on Kubernetes clusters using various container runtimes.

## Installation

*   Kubeflow Pipelines can be installed as part of the [Kubeflow Platform](https://www.kubeflow.org/docs/started/installing-kubeflow/#kubeflow-platform).
*   Alternatively, deploy [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/) as a standalone service.

## Documentation

*   **Getting Started:** [Kubeflow Pipelines overview](https://www.kubeflow.org/docs/components/pipelines/overview/)
*   **SDK Usage:** [Using the Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **API Reference:** [Pipelines API doc](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
*   **Python SDK:** [Python SDK reference docs](https://kubeflow-pipelines.readthedocs.io/en/stable/)
*   **AI-Powered Documentation:** Check out the AI-generated repo documentation on [DeepWiki](https://deepwiki.com/kubeflow/pipelines).  *Note: This is AI generated and may not have completely accurate information.*

## Contributing

Contribute to Kubeflow Pipelines by following the guidelines in [How to Contribute](./CONTRIBUTING.md).
To build and deploy from source, see the [developer guide](./developer_guide.md).

## Community

*   **Community Meetings:** Every other Wed 10-11AM (PST).
    *   [Calendar Invite](https://calendar.google.com/event?action=TEMPLATE&tmeid=NTdoNG5uMDBtcnJlYmdlOWt1c2lkY25jdmlfMjAxOTExMTNUMTgwMDAwWiBqZXNzaWV6aHVAZ29vZ2xlLmNvbQ&tmsrc=jessiezhu%40google.com&scp=ALL)
    *   [Direct Meeting Link](https://zoom.us/j/92607298595?pwd%3DVlKLUbiguGkbT9oKbaoDmCxrhbRop7.1&sa=D&source=calendar&ust=1736264977415448&usg=AOvVaw1EIkjFsKy0d4yQPptIJS3x)
    *   [Meeting notes](http://bit.ly/kfp-meeting-notes)
*   **Slack:** Find us in the #kubeflow-pipelines channel on the Cloud Native Computing Foundation Slack workspace. More details at [https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

## Architecture

*   Learn more about the KFP Architecture at [Architecture.md](docs/Architecture.md)

## Blog Posts

*   [From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle with Kubeflow](https://blog.kubeflow.org/fraud-detection-e2e/) (By [Helber Belmiro](https://github.com/hbelmiro))
*   [Getting started with Kubeflow Pipelines](https://cloud.google.com/blog/products/ai-machine-learning/getting-started-kubeflow-pipelines) (By Amy Unruh)
*   How to create and deploy a Kubeflow Machine Learning Pipeline (By Lak Lakshmanan)
    *   [Part 1: How to create and deploy a Kubeflow Machine Learning Pipeline](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
    *   [Part 2: How to deploy Jupyter notebooks as components of a Kubeflow ML pipeline](https://medium.com/data-science/how-to-deploy-jupyter-notebooks-as-components-of-a-kubeflow-ml-pipeline-part-2-b1df77f4e5b3)
    *   [Part 3: How to carry out CI/CD in Machine Learning (“MLOps”) using Kubeflow ML pipelines](https://medium.com/google-cloud/how-to-carry-out-ci-cd-in-machine-learning-mlops-using-kubeflow-ml-pipelines-part-3-bdaf68082112)

## Acknowledgments

Kubeflow Pipelines leverages [Argo Workflows](https://github.com/argoproj/argo-workflows) for orchestration, and we appreciate their contributions.