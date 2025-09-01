# KServe: Production-Ready Model Serving on Kubernetes

**KServe** is an open-source, cloud-agnostic platform that simplifies deploying and managing machine learning (ML) models on Kubernetes for production. [(See the original repo here)](https://github.com/kserve/kserve)

## Key Features

*   **Standardized Inference:** Supports standardized inference protocols across various ML frameworks, including TensorFlow, XGBoost, PyTorch, and Hugging Face Transformers/LLMs, with OpenAI specification for generative models.
*   **Serverless Inference Workloads:** Enables serverless inference with request-based autoscaling, including scale-to-zero, optimized for both CPU and GPU resources.
*   **High Scalability and Density:** Leverages ModelMesh for high-scale, high-density, and frequently-changing model serving deployments.
*   **Complete ML Serving Solution:** Provides a simple, pluggable platform for inference, pre/post-processing, monitoring, and explainability.
*   **Advanced Deployment Capabilities:** Supports advanced deployment strategies like canary rollouts, pipelines, and ensembles via InferenceGraph.
*   **Kubernetes Native:** Built as a Kubernetes Custom Resource Definition (CRD), integrates seamlessly with Kubernetes infrastructure.
*   **Cloud-Agnostic:** Designed to run on any Kubernetes cluster, regardless of cloud provider.
*   **Model Mesh:** Enables advanced features like model discovery, versioning, and routing.

## Why Choose KServe?

KServe offers a comprehensive solution for serving both predictive and generative AI models in production environments, designed for scalability, ease of use, and flexibility. It abstracts away the complexities of model serving, providing a streamlined experience with features like autoscaling, health checks, and advanced deployment strategies.

## Installation

KServe offers multiple installation options to fit your needs:

*   **Serverless Installation:** Leverage Knative for serverless deployments.
*   **Raw Kubernetes Deployment:** A lightweight installation option without serverless capabilities.
*   **ModelMesh Installation:** Enables high-scale, high-density model serving.
*   **Quick Installation:** Easily set up KServe on your local machine.
*   **Kubeflow Installation:** Integrated as an essential addon component within Kubeflow, consult the Kubeflow KServe documentation for more details.

## Get Started

*   **Create Your First InferenceService:** [Follow this guide](https://kserve.github.io/website/docs/getting-started/genai-first-isvc) to deploy your first model.
*   **Learn More:** Visit the [KServe website](https://kserve.github.io/website/) for detailed documentation and community resources.
*   **Roadmap:** Explore future development plans in the [Roadmap](./ROADMAP.md).
*   **API Reference:** Access the [InferenceService API Reference](https://kserve.github.io/website/docs/reference/crd-api).
*   **Developer Guide:** Learn about KServe development in the [Developer Guide](https://kserve.github.io/website/docs/developer-guide).
*   **Contributor Guide:** Find resources for contributing to KServe in the [Contributor Guide](https://kserve.github.io/website/docs/developer-guide/contribution).
*   **Adopters:** See which organizations are using KServe at the [Adopters page](https://kserve.github.io/website/docs/community/adopters).